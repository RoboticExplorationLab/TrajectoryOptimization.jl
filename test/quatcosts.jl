using Test
using TrajectoryOptimization
using RobotDynamics
using RobotDynamics: LieState
import RobotZoo.Quadrotor
using BenchmarkTools
using Rotations
using StaticArrays, LinearAlgebra, ForwardDiff
const TO = TrajectoryOptimization

for rot in (:UnitQuaternion, :MRP, :RodriguesParam)
    @eval rotation_name(::Type{<:$rot}) = $rot
end

struct DiagonalLieCost{n,m,T,nV,nR,Rot} <: TO.QuadraticCostFunction{n,m,T}
    Q::SVector{nV,T}
    R::Diagonal{T,SVector{m,T}}
    q::SVector{nV,T}
    r::SVector{m,T}
    c::T
    w::Vector{T}                     # weights on rotations (1 per rotation)
    vinds::SVector{nV,Int}           # inds of vector states
    qinds::Vector{SVector{nR,Int}}   # inds of rot states
    qrefs::Vector{UnitQuaternion{T}} # reference rotations
    function DiagonalLieCost(s::LieState{Rot,P},
            Q::AbstractVector{<:Real}, R::AbstractVector{<:Real}, 
            q::AbstractVector{<:Real}, r::AbstractVector{<:Real},
            c::Real, w::AbstractVector,
            qrefs::Vector{<:Rotation}
        ) where {Rot,P}
        n = length(s)
        m = length(R)
        nV = sum(P)
        nR = Rotations.params(Rot)
        num_rots = length(P)-1
        @assert length(Q) == length(q) == nV
        @assert length(r) == m 
        @assert length(qrefs) == length(w) == num_rots
        vinds = [RobotDynamics.vec_inds(Rot, P, i) for i = 1:num_rots+1]
        rinds = [RobotDynamics.rot_inds(Rot, P, i) for i = 1:num_rots]
        vinds = SVector{nV}(vcat(vinds...))
        rinds = SVector{nR}.(rinds)
        qrefs = UnitQuaternion.(qrefs)
        T = promote_type(eltype(Q), eltype(R), eltype(q), eltype(r), typeof(c), eltype(w))
        R = Diagonal(SVector{m}(R))
        new{n,m,T,nV,nR,rotation_name(Rot)}(Q, R, q, r, c, w, vinds, rinds, qrefs)
    end
end
Union{<:Diagonal{Tr},<:AbstractVector{Tr}} where {Tr}

function DiagonalLieCost(s::LieState{Rot,P}, 
        Q::Vector{<:AbstractVector{Tq}}, R::Union{<:Diagonal{Tr},<:AbstractVector{Tr}};
        q = [zeros(Tq,p) for p in P], r=zeros(Tr,size(R,1)), 
        c=0.0, w=ones(RobotDynamics.num_rotations(s)),
        qrefs=[one(UnitQuaternion) for i in 1:RobotDynamics.num_rotations(s)]
    ) where {Rot,P,Tq,Tr}
    @assert length.(Q) == collect(P) == length.(q)
end
P = (3,6)
Qs = [rand(p) for p in P]
s = LieState(UnitQuaternion,3,6)
R = rand(4)
DiagonalLieCost(s, Qs, R)

RobotDynamics.rotation_type(::DiagonalLieCost{<:Any,<:Any,<:Any, <:Any,<:Any, Rot}) where Rot = Rot
RobotDynamics.state_dim(::DiagonalLieCost{n}) where n = n
RobotDynamics.control_dim(::DiagonalLieCost{<:Any,m}) where m = m
TO.is_blockdiag(::DiagonalLieCost) = true
TO.is_diag(::DiagonalLieCost) = true

function TO.stage_cost(cost::DiagonalLieCost, x::AbstractVector)
    Rot = RobotDynamics.rotation_type(cost)
    Jv = veccost(cost.Q, cost.q, x, cost.vinds)
    Jr = quatcost(Rot, cost.w, x, cost.qinds, cost.qrefs)
    return Jv + Jr
end

function veccost(Q, q, x, vinds)
    xv = x[vinds]
    0.5*xv'Diagonal(Q)*xv + q'xv
end

function quatcost(::Type{Rot}, w, x, qinds, qref) where Rot<:Rotation
    J = zero(eltype(x))
    for i = 1:length(qinds) 
        # q = Rotations.params(UnitQuaternion(Rot(x[qinds[i]])))
        q = toquat(Rot, x[qinds[i]])
        qd = Rotations.params(qref[i])
        err = q'qd
        J = w[i]*min(1-err,1+err)
    end
    return J
end

function TO.gradient!(E::TO.QuadraticCostFunction, cost::DiagonalLieCost, x::AbstractVector)
    # Vector states
    Rot = RobotDynamics.rotation_type(cost)
    xv = x[cost.vinds]
    E.q[cost.vinds] .= cost.Q .* xv + cost.q

    # Quaternion states
    for i = 1:length(cost.qinds)
        qind = cost.qinds[i]
        q = toquat(Rot, x[qind])
        qref = Rotations.params(cost.qrefs[i])
        dq = q'qref
        E.q[qind] .= -cost.w[i]*qref*sign(dq)
    end
    return false
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::DiagonalLieCost{n}, x::AbstractVector) where n
    for (i,j) in enumerate(cost.vinds) 
        E.Q[j,j] = cost.Q[i]
    end
end

toquat(::Type{<:UnitQuaternion}, q::AbstractVector) = q
toquat(::Type{Rot}, q::AbstractVector) where Rot <: Rotation = 
    Rotations.params(UnitQuaternion(Rot(q),false))

## Set up
model = Quadrotor{UnitQuaternion}()
x, u = rand(model)
s = LieState(model)
Q = rand(RobotDynamics.state_dim_vec(s))
R = rand(control_dim(model))
q = rand(RobotDynamics.state_dim_vec(s))
r = rand(control_dim(model))
c = rand()
w = rand(RobotDynamics.num_rotations(s))
qrefs = [rand(UnitQuaternion) for i = 1:RobotDynamics.num_rotations(s)]

## Call inner constructor
costfun = DiagonalLieCost(s, Q, R, q, r, c, w, qrefs)

# Test stage cost
TO.stage_cost(costfun, x)
p,quat,v,ω = RobotDynamics.parse_state(model, x)
Qr,Qv,Qω = Diagonal.((Q[1:3],Q[4:6],Q[7:9]))
Jv = 0.5*(p'Qr*p + v'Qv*v + ω'Qω*ω) + q[1:3]'p + q[4:6]'v + q[7:9]'ω
q0 = Rotations.params(quat)
qref = Rotations.params(qrefs[1])
dq = q0'qref
Jr = w[1]*min(1-dq,1+dq)
@test Jr + Jv ≈ TO.stage_cost(costfun, x)

Ju = 0.5*u'Diagonal(R)*u + r'u
@test TO.stage_cost(costfun, x, u) ≈ Jr + Jv + Ju

## Gradient
grad = ForwardDiff.gradient(x->TO.stage_cost(costfun, x), x)
gradq = -qref*w[1]*sign(dq)
@test grad ≈ [Qr*p + q[1:3]; gradq; Qv*v + q[4:6] ; Qω*ω + q[7:9]]
vinds = costfun.vinds
grad2 = zeros(length(grad))
grad2[vinds] .= Diagonal(Q)*x[vinds] + q

E = QuadraticCost{Float64}(13,4)
TO.gradient!(E, costfun, x, u)
@test E.q ≈ grad
@test E.r ≈ Diagonal(R)*u + r

## Hessian
hess = ForwardDiff.hessian(x->TO.stage_cost(costfun, x), x)
hess2 = zero(grad2)
hess2[vinds] .= Q
@test Diagonal(hess2) ≈ hess

E.Q .*= 0
@test TO.hessian!(E, costfun, x, u) == true
@test E.Q ≈ hess
@test E.R ≈ Diagonal(R)