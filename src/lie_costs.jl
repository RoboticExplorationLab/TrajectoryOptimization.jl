for rot in (:UnitQuaternion, :MRP, :RodriguesParam)
    @eval rotation_name(::Type{<:$rot}) = $rot
end

struct DiagonalLieCost{n,m,T,nV,nR,Rot} <: QuadraticCostFunction{n,m,T}
    Q::SVector{nV,T}
    R::Diagonal{T,SVector{m,T}}
    q::SVector{nV,T}
    r::SVector{m,T}
    c::T
    w::Vector{T}                     # weights on rotations (1 per rotation)
    vinds::SVector{nV,Int}           # inds of vector states
    qinds::Vector{SVector{nR,Int}}   # inds of rot states
    qrefs::Vector{UnitQuaternion{T}} # reference rotations
    function DiagonalLieCost(s::RD.LieState{Rot,P},
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
    function DiagonalLieCost{Rot}(Q,R,q,r,c,w,vinds,qinds,qrefs) where Rot
        nV = length(Q)
        num_rots = length(qinds)
        nR = length(qinds[1])
        n = nV + num_rots*nR
        m = size(R,1)
        T = eltype(Q)
        new{n,m,T,nV,nR,rotation_name(Rot)}(Q, R, q, r, c, w, vinds, qinds, qrefs)
    end
end

function Base.copy(c::DiagonalLieCost)
    Rot = RD.rotation_type(c)
    DiagonalLieCost{Rot}(copy(c.Q), c.R, copy(c.q), copy(c.r), c.c, copy(c.w), 
        copy(c.vinds), copy(c.qinds), deepcopy(c.qrefs)
    )
end

function DiagonalLieCost(s::RD.LieState{Rot,P}, 
        Q::Vector{<:AbstractVector{Tq}}, 
        R::Union{<:Diagonal{Tr},<:AbstractVector{Tr}};
        q = [zeros(Tq,p) for p in P], 
        r=zeros(Tr,size(R,1)), 
        c=0.0, 
        w=ones(RobotDynamics.num_rotations(s)),
        qrefs=[one(UnitQuaternion) for i in 1:RobotDynamics.num_rotations(s)]
    ) where {Rot,P,Tq,Tr}
    @assert length.(Q) == collect(P) == length.(q)
    Q = vcat(Q...)
    q = vcat(q...)
    if R isa Diagonal
        R = R.diag
    end
    DiagonalLieCost(s, Q, R, q, r, c, w, qrefs)
end

function DiagonalLieCost(s::RD.LieState{Rot,P}, Q::Diagonal, R::Diagonal; 
        q::AbstractVector=zero(Q.diag), kwargs...
    ) where {Rot,P}
    if length(Q.diag) == sum(P)
        vinds = cumsum(insert(SVector(P), 1, 0))
        vinds = [vinds[i]+1:vinds[i+1] for i = 1:length(P)]
        Qs = [Q.diag[vind] for vind in vinds]
        qs = [q[vind] for vind in vinds]
        DiagonalLieCost(s, Qs, R.diag; q=qs, kwargs...)
    else
        num_rots = length(P) - 1
        vinds = [RobotDynamics.vec_inds(Rot, P, i) for i = 1:num_rots+1]
        rinds = [RobotDynamics.rot_inds(Rot, P, i) for i = 1:num_rots]

        # Split Q and q into vector and rotation components
        Qv = [Q[vind] for vind in vinds]
        Qr = [Q[rind] for rind in rinds]
        qv = [q[vind] for vind in vinds]

        # Use sum of diagonal Q as w
        w = sum.(Qr)
        @show vinds

        DiagonalLieCost(s, Qv, R.diag; q=qv, w=w, kwargs...)
    end
end

function LieLQRCost(s::RD.LieState{Rot,P}, 
        Q::Vector{<:AbstractVector}, 
        R::Union{<:AbstractVector,<:Diagonal},
        xf; 
        w=ones(length(Q)-1), 
        uf=zeros(size(R,1))
    ) where {Rot,P}
    if R isa AbstractVector
        R = Diagonal(R)
    end
    num_rots = length(P) - 1
    vinds = [RobotDynamics.vec_inds(Rot, P, i) for i = 1:num_rots+1]
    qinds = [RobotDynamics.rot_inds(Rot, P, i) for i = 1:num_rots]
    qrefs = [Rot(xf[qinds[i]]) for i = 1:num_rots]
    q = [-Diagonal(Q[i])*xf[vinds[i]] for i = 1:length(P)] 
    r = -R*uf
    c = sum(0.5*xf[vinds[i]]'Diagonal(Q[i])*xf[vinds[i]] for i = 1:length(P))
    c += 0.5*uf'R*uf
    return DiagonalLieCost(s, Q, R; q=q, r=r, c=c, w=w, qrefs=qrefs)
end

RobotDynamics.rotation_type(::DiagonalLieCost{<:Any,<:Any,<:Any, <:Any,<:Any, Rot}) where Rot = Rot
RobotDynamics.state_dim(::DiagonalLieCost{n}) where n = n
RobotDynamics.control_dim(::DiagonalLieCost{<:Any,m}) where m = m
is_blockdiag(::DiagonalLieCost) = true
is_diag(::DiagonalLieCost) = true


function stage_cost(cost::DiagonalLieCost, x::AbstractVector)
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

function gradient!(E::QuadraticCostFunction, cost::DiagonalLieCost, x::AbstractVector)
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
        jac = Rotations.jacobian(UnitQuaternion, Rot(q))
        E.q[qind] .= -cost.w[i]*sign(dq)*jac'qref
    end
    return false
end

function hessian!(E::QuadraticCostFunction, cost::DiagonalLieCost{n}, x::AbstractVector) where n
    for (i,j) in enumerate(cost.vinds) 
        E.Q[j,j] = cost.Q[i]
    end
    return true
end

toquat(::Type{<:UnitQuaternion}, q::AbstractVector) = q
toquat(::Type{Rot}, q::AbstractVector) where Rot <: Rotation = 
    Rotations.params(UnitQuaternion(Rot(q)))


# Jacobians missing from Rotatations
function Rotations.jacobian(::Type{UnitQuaternion}, p::RodriguesParam)
    p = Rotations.params(p)
    np = p'p 
    d = 1/sqrt(1 + p'p)
    d3 = d*d*d
    ds = -d3 * p
    pp = -d3*(p*p')
    SA[
        ds[1] ds[2] ds[3];
        pp[1] + d pp[4] pp[7]; 
        pp[2] pp[5] + d pp[8];
        pp[3] pp[6] pp[9] + d;
    ]
end

function Rotations.jacobian(::Type{RodriguesParam}, q::UnitQuaternion)
    s = 1/q.w
    SA[
        -s*s*q.x s 0 0;
        -s*s*q.y 0 s 0;
        -s*s*q.z 0 0 s
    ]
end

Rotations.jacobian(::Type{R}, q::R) where R = I