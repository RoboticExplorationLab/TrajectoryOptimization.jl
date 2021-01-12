for rot in (:UnitQuaternion, :MRP, :RodriguesParam)
    @eval rotation_name(::Type{<:$rot}) = $rot
end

# struct DiagonalLieCost{n,m,T,nV,nR,Rot} <: QuadraticCostFunction{n,m,T}
#     Q::SVector{nV,T}
#     R::Diagonal{T,SVector{m,T}}
#     q::SVector{nV,T}
#     r::SVector{m,T}
#     c::T
#     w::Vector{T}                     # weights on rotations (1 per rotation)
#     vinds::SVector{nV,Int}           # inds of vector states
#     qinds::Vector{SVector{nR,Int}}   # inds of rot states
#     qrefs::Vector{UnitQuaternion{T}} # reference rotations
#     function DiagonalLieCost(s::RD.LieState{Rot,P},
#             Q::AbstractVector{<:Real}, R::AbstractVector{<:Real}, 
#             q::AbstractVector{<:Real}, r::AbstractVector{<:Real},
#             c::Real, w::AbstractVector,
#             qrefs::Vector{<:Rotation}
#         ) where {Rot,P}
#         n = length(s)
#         m = length(R)
#         nV = sum(P)
#         nR = Rotations.params(Rot)
#         num_rots = length(P)-1
#         @assert length(Q) == length(q) == nV
#         @assert length(r) == m 
#         @assert length(qrefs) == length(w) == num_rots
#         vinds = [RobotDynamics.vec_inds(Rot, P, i) for i = 1:num_rots+1]
#         rinds = [RobotDynamics.rot_inds(Rot, P, i) for i = 1:num_rots]
#         vinds = SVector{nV}(vcat(vinds...))
#         rinds = SVector{nR}.(rinds)
#         qrefs = UnitQuaternion.(qrefs)
#         T = promote_type(eltype(Q), eltype(R), eltype(q), eltype(r), typeof(c), eltype(w))
#         R = Diagonal(SVector{m}(R))
#         new{n,m,T,nV,nR,rotation_name(Rot)}(Q, R, q, r, c, w, vinds, rinds, qrefs)
#     end
#     function DiagonalLieCost{Rot}(Q,R,q,r,c,w,vinds,qinds,qrefs) where Rot
#         nV = length(Q)
#         num_rots = length(qinds)
#         nR = length(qinds[1])
#         n = nV + num_rots*nR
#         m = size(R,1)
#         T = eltype(Q)
#         new{n,m,T,nV,nR,rotation_name(Rot)}(Q, R, q, r, c, w, vinds, qinds, qrefs)
#     end
# end

# function Base.copy(c::DiagonalLieCost)
#     Rot = RD.rotation_type(c)
#     DiagonalLieCost{Rot}(copy(c.Q), c.R, copy(c.q), copy(c.r), c.c, copy(c.w), 
#         copy(c.vinds), copy(c.qinds), deepcopy(c.qrefs)
#     )
# end

# function DiagonalLieCost(s::RD.LieState{Rot,P}, 
#         Q::Vector{<:AbstractVector{Tq}}, 
#         R::Union{<:Diagonal{Tr},<:AbstractVector{Tr}};
#         q = [zeros(Tq,p) for p in P], 
#         r=zeros(Tr,size(R,1)), 
#         c=0.0, 
#         w=ones(RobotDynamics.num_rotations(s)),
#         qrefs=[one(UnitQuaternion) for i in 1:RobotDynamics.num_rotations(s)]
#     ) where {Rot,P,Tq,Tr}
#     @assert length.(Q) == collect(P) == length.(q)
#     Q = vcat(Q...)
#     q = vcat(q...)
#     if R isa Diagonal
#         R = R.diag
#     end
#     if w isa Real
#         w = fill(w,length(qrefs))
#     end
#     DiagonalLieCost(s, Q, R, q, r, c, w, qrefs)
# end

# function DiagonalLieCost(s::RD.LieState{Rot,P}, Q::Diagonal, R::Diagonal; 
#         q::AbstractVector=zero(Q.diag), kwargs...
#     ) where {Rot,P}
#     if length(Q.diag) == sum(P)
#         vinds = cumsum(insert(SVector(P), 1, 0))
#         vinds = [vinds[i]+1:vinds[i+1] for i = 1:length(P)]
#         Qs = [Q.diag[vind] for vind in vinds]
#         qs = [q[vind] for vind in vinds]
#         DiagonalLieCost(s, Qs, R.diag; q=qs, kwargs...)
#     else
#         num_rots = length(P) - 1
#         vinds = [RobotDynamics.vec_inds(Rot, P, i) for i = 1:num_rots+1]
#         rinds = [RobotDynamics.rot_inds(Rot, P, i) for i = 1:num_rots]

#         # Split Q and q into vector and rotation components
#         Qv = [Q[vind] for vind in vinds]
#         Qr = [Q[rind] for rind in rinds]
#         qv = [q[vind] for vind in vinds]

#         # Use sum of diagonal Q as w
#         w = sum.(Qr)
#         @show vinds

#         DiagonalLieCost(s, Qv, R.diag; q=qv, w=w, kwargs...)
#     end
# end

# function LieLQRCost(s::RD.LieState{Rot,P}, 
#         Q::Diagonal,
#         R::Diagonal,
#         xf, uf=zeros(size(R,1)); kwargs...) where {Rot,P}
#     @assert length(Q.diag) == length(s)
#     vinds = [RobotDynamics.vec_inds(Rot, P, i) for i = 1:length(P)]
#     Qs = [Q[vind] for vind in vinds]
#     LieLQRCost(s, Qs, R, xf, uf; kwargs...)
# end

# function LieLQRCost(s::RD.LieState{Rot,P}, 
#         Q::Vector{<:AbstractVector}, 
#         R::Union{<:AbstractVector,<:Diagonal},
#         xf, 
#         uf=zeros(size(R,1));
#         w=ones(length(Q)-1), 
#     ) where {Rot,P}
#     if R isa AbstractVector
#         R = Diagonal(R)
#     end
#     num_rots = length(P) - 1
#     vinds = [RobotDynamics.vec_inds(Rot, P, i) for i = 1:num_rots+1]
#     qinds = [RobotDynamics.rot_inds(Rot, P, i) for i = 1:num_rots]
#     qrefs = [Rot(xf[qinds[i]]) for i = 1:num_rots]
#     q = [-Diagonal(Q[i])*xf[vinds[i]] for i = 1:length(P)] 
#     r = -R*uf
#     c = sum(0.5*xf[vinds[i]]'Diagonal(Q[i])*xf[vinds[i]] for i = 1:length(P))
#     c += 0.5*uf'R*uf
#     return DiagonalLieCost(s, Q, R; q=q, r=r, c=c, w=w, qrefs=qrefs)
# end

# RobotDynamics.rotation_type(::DiagonalLieCost{<:Any,<:Any,<:Any, <:Any,<:Any, Rot}) where Rot = Rot
# RobotDynamics.state_dim(::DiagonalLieCost{n}) where n = n
# RobotDynamics.control_dim(::DiagonalLieCost{<:Any,m}) where m = m
# is_blockdiag(::DiagonalLieCost) = true
# is_diag(::DiagonalLieCost) = true


# function stage_cost(cost::DiagonalLieCost, x::AbstractVector)
#     Rot = RobotDynamics.rotation_type(cost)
#     Jv = veccost(cost.Q, cost.q, x, cost.vinds)
#     Jr = quatcost(Rot, cost.w, x, cost.qinds, cost.qrefs)
#     return Jv + Jr
# end

# function veccost(Q, q, x, vinds)
#     xv = x[vinds]
#     0.5*xv'Diagonal(Q)*xv + q'xv
# end

# function quatcost(::Type{Rot}, w, x, qinds, qref) where Rot<:Rotation
#     J = zero(eltype(x))
#     for i = 1:length(qinds) 
#         # q = Rotations.params(UnitQuaternion(Rot(x[qinds[i]])))
#         q = toquat(Rot, x[qinds[i]])
#         qd = Rotations.params(qref[i])
#         err = q'qd
#         J = w[i]*min(1-err,1+err)
#     end
#     return J
# end

# function gradient!(E::QuadraticCostFunction, cost::DiagonalLieCost, x::AbstractVector)
#     # Vector states
#     Rot = RobotDynamics.rotation_type(cost)
#     xv = x[cost.vinds]
#     E.q[cost.vinds] .= cost.Q .* xv + cost.q

#     # Quaternion states
#     for i = 1:length(cost.qinds)
#         qind = cost.qinds[i]
#         q = toquat(Rot, x[qind])
#         qref = Rotations.params(cost.qrefs[i])
#         dq = q'qref
#         jac = Rotations.jacobian(UnitQuaternion, Rot(q))
#         E.q[qind] .= -cost.w[i]*sign(dq)*jac'qref
#     end
#     return false
# end

# function hessian!(E::QuadraticCostFunction, cost::DiagonalLieCost{n}, x::AbstractVector) where n
#     for (i,j) in enumerate(cost.vinds) 
#         E.Q[j,j] = cost.Q[i]
#     end
#     return true
# end

# toquat(::Type{<:UnitQuaternion}, q::AbstractVector) = q
# toquat(::Type{Rot}, q::AbstractVector) where Rot <: Rotation = 
#     Rotations.params(UnitQuaternion(Rot(q)))


# # Jacobians missing from Rotatations
# function Rotations.jacobian(::Type{UnitQuaternion}, p::RodriguesParam)
#     p = Rotations.params(p)
#     np = p'p 
#     d = 1/sqrt(1 + p'p)
#     d3 = d*d*d
#     ds = -d3 * p
#     pp = -d3*(p*p')
#     SA[
#         ds[1] ds[2] ds[3];
#         pp[1] + d pp[4] pp[7]; 
#         pp[2] pp[5] + d pp[8];
#         pp[3] pp[6] pp[9] + d;
#     ]
# end

# function Rotations.jacobian(::Type{RodriguesParam}, q::UnitQuaternion)
#     s = 1/q.w
#     SA[
#         -s*s*q.x s 0 0;
#         -s*s*q.y 0 s 0;
#         -s*s*q.z 0 0 s
#     ]
# end

# Rotations.jacobian(::Type{R}, q::R) where R = I



############################################################################################
#                        QUADRATIC QUATERNION COST FUNCTION
############################################################################################
"""
    DiagonalQuatCost

Quadratic cost function for states that includes a 3D rotation, that penalizes deviations 
    from a provided 3D rotation, represented as a Unit Quaternion.

The cost function penalizes geodesic distance between unit quaternions:

``\\frac{1}{2} \\big( x^T Q x + u^T R u \\big) + q^T x + r^T u + c + w \\min 1 \\pm p_f^T p``

where ``p`` is the quaternion extracted from ``x`` (i.e. `p = x[q_ind]`), and ``p_f`` 
is the reference quaternion. ``Q`` and ``R`` are assumed to be diagonal.

We've found this perform better than penalizing a quadratic on the quaternion error 
state ([`ErrorQuadratic`](@ref)). This cost should still be considered experimental.

# Constructors
* `DiagonalQuatCost(Q::Diagonal, R::Diagonal, q, r, c, w, q_ref, q_ind; terminal)`
* `DiagonalQuatCost(Q::Diagonal, R::Diagonal; q, r, c, w, q_ref, q_ind, terminal)`

where `q_ref` is the reference quaternion (provided as a `SVector{4}`), and 
    `q_ind::SVector{4,Int}` provides the indices of the quaternion in the state vector 
    (default = `SA[4,5,6,7]`). Note that `Q` and `q` are the size of the full state, 
    so `Q.diag[q_ind]` and `q[qind]` should typically be zero.
"""
struct DiagonalQuatCost{N,M,T,N4} <: QuadraticCostFunction{N,M,T}
    Q::Diagonal{T,SVector{N,T}}
    R::Diagonal{T,SVector{M,T}}
    q::SVector{N,T}
    r::SVector{M,T}
    c::T
    w::T
    q_ref::SVector{4,T}
    q_ind::SVector{4,Int}
    Iq::SMatrix{N,4,T,N4}
    terminal::Bool
    function DiagonalQuatCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
            q::SVector{N,T}, r::SVector{M,T}, c::T, w::T,
            q_ref::SVector{4,T}, q_ind::SVector{4,Int}; terminal::Bool=false) where {T,N,M}
        Iq = @MMatrix zeros(N,4)
        for i = 1:4
            Iq[q_ind[i],i] = 1
        end
        Iq = SMatrix{N,4}(Iq)
        return new{N,M,T,N*4}(Q, R, q, r, c, w, q_ref, q_ind, Iq, terminal)
    end
end

state_dim(::DiagonalQuatCost{N,M,T}) where {T,N,M} = N
control_dim(::DiagonalQuatCost{N,M,T}) where {T,N,M} = M
is_blockdiag(::DiagonalQuatCost) = true
is_diag(::DiagonalQuatCost) = true

function DiagonalQuatCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}};
        q=(@SVector zeros(N)), r=(@SVector zeros(M)), c=zero(T), w=one(T),
        q_ref=(@SVector [1.0,0,0,0]), q_ind=(@SVector [4,5,6,7])) where {T,N,M}
    DiagonalQuatCost(Q, R, q, r, c, q_ref, q_ind)
end

function stage_cost(cost::DiagonalQuatCost, x::SVector, u::SVector)
    stage_cost(cost, x) + 0.5*u'cost.R*u + cost.r'u
end

function stage_cost(cost::DiagonalQuatCost, x::SVector)
    J = 0.5*x'cost.Q*x + cost.q'x + cost.c
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    J += cost.w*min(1+dq, 1-dq)
end

function gradient!(E, cost::DiagonalQuatCost, z::AbstractKnotPoint, cache=nothing)
    x,u = state(z), control(z)
    Qx = cost.Q*x + cost.q
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    if dq < 0
        Qx += cost.w*cost.Iq*cost.q_ref
    else
        Qx -= cost.w*cost.Iq*cost.q_ref
    end
    E.q .= Qx
    if !is_terminal(z)
        E.r .= cost.R*u .+ cost.r
    end
    return false
end

"""
    QuatLQRCost(Q, R, xf, [uf; w, quat_ind])

Defines a cost function of the form:

``\\frac{1}{2} \\big( (x - x_f)^T Q (x - x_f) + (u - u_f)^T R (u - u_f) \\big) + w \\min 1 \\pm q_f^T q``

where ``Q`` and ``R`` are diagonal, ``x_f`` is the goal state, 
``u_f`` is the reference control, and ``q_f``, ``q`` are
the quaternions, extracted from ``x`` using `quat_ind`, i.e. `q = x[quat_ind]`.

The last term is the geodesic distance between quaternions. It's typically recommended that 
`Q.diag[quad_ind] == zeros(4)`.

This is just a convenience constructor for [`DiagonalQuatCost`](@ref).

# Example 
For a standard rigid body state vector `x = [p; q; v; ω]`, where `q` is a unit quaternion,
we could define a cost function that penalizes the distance to the goal state `xf`. 
We can create this cost function as follows:
```julia
Q = Diagonal(SVector(RBState(fill(0.1,3), zeros(4), fill(0.1,3), fill(0.1,3))))
R = Diagonal(@SVector fill(0.01, 6))
xf = RBState([1,2,3], rand(UnitQuaternion), zeros(3), zeros(3))
QuatLQRCost(Q,R,xf)
```
We can add a reference control and change the weight on the rotation error with the optional
arguments:
```julia
QuatLQRCost(Q,R,xf,uf, w=10.0)
```
which is equivalent to
```julia
QuatLQRCost(Q,R,xf,uf, w=10.0, quat_inds=4:7)
```
"""
function QuatLQRCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}}, xf,
        uf=(@SVector zeros(M)); w=one(T), quat_ind=(@SVector [4,5,6,7])) where {T,N,M}
    @assert length(quat_ind) == 4 "quat_ind argument must be of length 4"
    quat_ind = SVector{4}(quat_ind)
    r = -R*uf
    q = -Q*xf
    c = 0.5*xf'Q*xf + 0.5*uf'R*uf
    q_ref = xf[quat_ind]
    return DiagonalQuatCost(Q, R, q, r, c, T(w), q_ref, quat_ind)
end

function change_dimension(cost::DiagonalQuatCost, n, m, ix, iu)
    Qd = zeros(n)
    Rd = zeros(m)
    q = zeros(n)
    r = zeros(m)
    Qd[ix] = diag(cost.Q)
    Rd[iu] = diag(cost.R)
    q[ix] = cost.q
    r[iu] = cost.r
    qind = (1:n)[ix[cost.q_ind]]
    DiagonalQuatCost(Diagonal(SVector{n}(Qd)), Diagonal(SVector{m}(Rd)), 
        SVector{n}(q), SVector{m}(r), cost.c, cost.w, cost.q_ref, qind)
end

function (+)(cost1::DiagonalQuatCost, cost2::QuadraticCostFunction)
    @assert state_dim(cost1) == state_dim(cost2)
    @assert control_dim(cost1) == control_dim(cost2)
    is_diag(cost2) || @assert norm(cost2.H) ≈ 0
    DiagonalQuatCost(cost1.Q + cost2.Q, cost1.R + cost2.R,
        cost1.q + cost2.q, cost1.r + cost2.r, cost1.c + cost2.c,
        cost1.w, cost1.q_ref, cost1.q_ind)
end

(+)(cost1::QuadraticCostFunction, cost2::DiagonalQuatCost) = cost2 + cost1

function Base.copy(c::DiagonalQuatCost)
    DiagonalQuatCost(c.Q, c.R, c.q, c.r, c.c, c.w, c.q_ref, c.q_ind)
end


############################################################################################
#                             Error Quadratic
############################################################################################

"""
    ErrorQuadratic{Rot,N,M}

Cost function of the form:

``\\frac{1}{2} (x_k \\ominus x_d)^T Q_k (x_k \\ominus x_d)``

where ``x_k \\ominus x_d`` is the error state, computed using 
`RobotDynamics.state_diff`. 
This cost function isn't recommended: we've found that `DiagonalQuatCost` usually
    peforms better and is much more computationally efficient.
"""
struct ErrorQuadratic{Rot,N,M} <: CostFunction
    model::RD.RigidBody{Rot}
    Q::Diagonal{Float64,SVector{12,Float64}}
    R::Diagonal{Float64,SVector{M,Float64}}
    r::SVector{M,Float64}
    c::Float64
    x_ref::SVector{N,Float64}
    q_ind::SVector{4,Int}
end
function Base.copy(c::ErrorQuadratic)
    ErrorQuadratic(c.model, c.Q, c.R, c.r, c.c, c.x_ref, c.q_ind)
end

state_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = N
control_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = M

function ErrorQuadratic(model::RD.RigidBody{Rot}, Q::Diagonal{T,<:SVector{N0}},
        R::Diagonal{T,<:SVector{M}},
        x_ref::SVector{N}, 
        u_ref=(@SVector zeros(T,M)); 
        r=(@SVector zeros(T,M)), 
        c=zero(T),
        q_ind=(@SVector [4,5,6,7])
    ) where {T,N,N0,M,Rot}
    if Rot <: UnitQuaternion && N0 == N 
        Qd = deleteat(Q.diag, 4)
        Q = Diagonal(Qd)
    end
    r += -R*u_ref
    c += 0.5*u_ref'R*u_ref
    return ErrorQuadratic{Rot,N,M}(model, Q, R, r, c, x_ref, q_ind)
end


function stage_cost(cost::ErrorQuadratic, x::AbstractVector)
    dx = RD.state_diff(cost.model, x, cost.x_ref, Rotations.CayleyMap())
    return 0.5*dx'cost.Q*dx + cost.c
end

function stage_cost(cost::ErrorQuadratic, x::AbstractVector, u::AbstractVector)
    stage_cost(cost, x) + 0.5*u'cost.R*u + cost.r'u
end

diffmethod(::ErrorQuadratic) = RobotDynamics.FiniteDifference()


# function gradient!(E, cost::ErrorQuadratic, z::AbstractKnotPoint, cache=nothing)
#     x,u = state(z), control(z)
#     f(x) = stage_cost(cost, x)
#     ForwardDiff.gradient!(E.q, f, x)
#     if !is_terminal(z)
#         E.u .= cost.R*u + cost.r
#     end
#     return false

#     model = cost.model
#     Q = cost.Q
#     q = RD.orientation(model, x)
#     q_ref = RD.orientation(model, cost.x_ref)
#     dq = Rotations.params(q_ref \ q)
#     err = RD.state_diff(model, x, cost.x_ref)
#     dx = @SVector [err[1],  err[2],  err[3],
#                     dq[1],   dq[2],   dq[3],   dq[4],
#                    err[7],  err[8],  err[9],
#                    err[10], err[11], err[12]]
#     # G = state_diff_jacobian(model, dx) # n × dn

#     # Gradient
#     dmap = inverse_map_jacobian(model, dx) # dn × n
#     # Qx = G'dmap'Q*err
#     Qx = dmap'Q*err
#     E.x .= Qx
#     if !is_terminal(z)
#         E.r .= cost.R * u
#     end
#     return false
# end
# # function gradient!(E::QuadraticCostFunction, cost::ErrorQuadratic, x, u)
# #     gradient!(E, cost, x)
# #     Qu = cost.R*u
# #     E.r .= Qu
# #     return false
# # end

# function hessian!(E, cost::ErrorQuadratic, z::AbstractKnotPoint, cache=nothing)
#     x,u = state(z), control(z)
#     f(x) = stage_cost(cost, x)
#     ForwardDiff.hessian!(E.Q, f, x)
#     return false

#     model = cost.model
#     Q = cost.Q
#     q = RD.orientation(model, x)
#     q_ref = RD.orientation(model, cost.x_ref)
#     dq = Rotations.params(q_ref\q)
#     err = RD.state_diff(model, x, cost.x_ref)
#     dx = @SVector [err[1],  err[2],  err[3],
#                     dq[1],   dq[2],   dq[3],   dq[4],
#                    err[7],  err[8],  err[9],
#                    err[10], err[11], err[12]]
#     # G = state_diff_jacobian(model, dx) # n × dn

#     # Gradient
#     dmap = inverse_map_jacobian(model, dx) # dn × n

#     # Hessian
#     ∇jac = inverse_map_∇jacobian(model, dx, Q*err)
#     # Qxx = G'dmap'Q*dmap*G + G'∇jac*G + ∇²differential(model, x, dmap'Q*err)
#     Qxx = dmap'Q*dmap + ∇jac #+ ∇²differential(model, x, dmap'Q*err)
#     E.Q = Qxx
#     E.H .*= 0 
#     if !is_terminal(z)
#         E.R .= cost.R
#     end
#     return false
# end

# function hessian!(E::QuadraticCostFunction, cost::ErrorQuadratic, x, u)
#     hessian!(E, cost, x)
#     E.R .= cost.R
#     return false
# end


function change_dimension(cost::ErrorQuadratic, n, m)
    n0,m0 = state_dim(cost), control_dim(cost)
    Q_diag = diag(cost.Q)
    R_diag = diag(cost.R)
    r = cost.r
    if n0 != n
        dn = n - n0  # assumes n > n0
        pad = @SVector zeros(dn) # assume the new states don't have quaternions
        Q_diag = [Q_diag; pad]
    end
    if m0 != m
        dm = m - m0  # assumes m > m0
        pad = @SVector zeros(dm)
        R_diag = [R_diag; pad]
        r = [r; pad]
    end
    ErrorQuadratic(cost.model, Diagonal(Q_diag), Diagonal(R_diag), r, cost.c,
        cost.x_ref, cost.q_ind)
end

function (+)(cost1::ErrorQuadratic, cost2::QuadraticCost)
    @assert control_dim(cost1) == control_dim(cost2)
    @assert norm(cost2.H) ≈ 0
    @assert norm(cost2.q) ≈ 0
    if state_dim(cost2) == 13
        rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
        Q2 = Diagonal(diag(cost2.Q)[rm_quat])
    else
        Q2 = cost2.Q
    end
    ErrorQuadratic(cost1.model, cost1.Q + Q2, cost1.R + cost2.R,
        cost1.r + cost2.r, cost1.c + cost2.c,
        cost1.x_ref, cost1.q_ind)
end

(+)(cost1::QuadraticCost, cost2::ErrorQuadratic) = cost2 + cost1

# @generated function RD.state_diff_jacobian(model::RD.RigidBody{<:UnitQuaternion},
#     x0::SVector{N,T}, errmap::D=Rotations.CayleyMap()) where {N,T,D}
#     if D <: IdentityMap
#         :(I)
#     else
#         quote
#             q0 = RD.orientation(model, x0)
#             # G = TrajectoryOptimization.∇differential(q0)
#             G = Rotations.∇differential(q0)
#             I1 = @SMatrix [1 0 0 0 0 0 0 0 0 0 0 0;
#                         0 1 0 0 0 0 0 0 0 0 0 0;
#                         0 0 1 0 0 0 0 0 0 0 0 0;
#                         0 0 0 G[1] G[5] G[ 9] 0 0 0 0 0 0;
#                         0 0 0 G[2] G[6] G[10] 0 0 0 0 0 0;
#                         0 0 0 G[3] G[7] G[11] 0 0 0 0 0 0;
#                         0 0 0 G[4] G[8] G[12] 0 0 0 0 0 0;
#                         0 0 0 0 0 0 1 0 0 0 0 0;
#                         0 0 0 0 0 0 0 1 0 0 0 0;
#                         0 0 0 0 0 0 0 0 1 0 0 0;
#                         0 0 0 0 0 0 0 0 0 1 0 0;
#                         0 0 0 0 0 0 0 0 0 0 1 0;
#                         0 0 0 0 0 0 0 0 0 0 0 1.]
#         end
#     end
# end
# function inverse_map_jacobian(model::RD.RigidBody{<:UnitQuaternion},
#     x::SVector, errmap=Rotations.CayleyMap())
#     q = RD.orientation(model, x)
#     # G = TrajectoryOptimization.inverse_map_jacobian(q)
#     G = Rotations.jacobian(inv(errmap), q)
#     return @SMatrix [
#             1 0 0 0 0 0 0 0 0 0 0 0 0;
#             0 1 0 0 0 0 0 0 0 0 0 0 0;
#             0 0 1 0 0 0 0 0 0 0 0 0 0;
#             0 0 0 G[1] G[4] G[7] G[10] 0 0 0 0 0 0;
#             0 0 0 G[2] G[5] G[8] G[11] 0 0 0 0 0 0;
#             0 0 0 G[3] G[6] G[9] G[12] 0 0 0 0 0 0;
#             0 0 0 0 0 0 0 1 0 0 0 0 0;
#             0 0 0 0 0 0 0 0 1 0 0 0 0;
#             0 0 0 0 0 0 0 0 0 1 0 0 0;
#             0 0 0 0 0 0 0 0 0 0 1 0 0;
#             0 0 0 0 0 0 0 0 0 0 0 1 0;
#             0 0 0 0 0 0 0 0 0 0 0 0 1;
#     ]
# end

# function inverse_map_∇jacobian(model::RD.RigidBody{<:UnitQuaternion},
#     x::SVector, b::SVector, errmap=Rotations.CayleyMap())
#     q = RD.orientation(model, x)
#     bq = @SVector [b[4], b[5], b[6]]
#     # ∇G = TrajectoryOptimization.inverse_map_∇jacobian(q, bq)
#     ∇G = Rotations.∇jacobian(inv(errmap), q, bq)
#     return @SMatrix [
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#         0 0 0 ∇G[1] ∇G[5] ∇G[ 9] ∇G[13] 0 0 0 0 0 0;
#         0 0 0 ∇G[2] ∇G[6] ∇G[10] ∇G[14] 0 0 0 0 0 0;
#         0 0 0 ∇G[3] ∇G[7] ∇G[11] ∇G[15] 0 0 0 0 0 0;
#         0 0 0 ∇G[4] ∇G[8] ∇G[12] ∇G[16] 0 0 0 0 0 0;
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0 0 0 0 0 0 0;
#     ]
# end



