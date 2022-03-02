for rot in (:UnitQuaternion, :MRP, :RodriguesParam)
    @eval rotation_name(::Type{<:$rot}) = $rot
end


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

function RD.evaluate(cost::DiagonalQuatCost, x, u)
    J = 0.5*x'cost.Q*x + cost.q'x + cost.c
    if !isempty(u)
        J += 0.5 * u'cost.R*u + cost.r'u
    end
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    J += cost.w*min(1+dq, 1-dq)
end

function gradient!(cost::DiagonalQuatCost{n,m}, grad, z::AbstractKnotPoint) where {n,m}
    x,u = state(z), control(z)
    ix,iu = 1:n, n+1:n+m
    x,u = state(z), control(z)
    Qx = cost.Q*x + cost.q
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    if dq < 0
        Qx += cost.w*cost.Iq*cost.q_ref
    else
        Qx -= cost.w*cost.Iq*cost.q_ref
    end
    grad[ix] .= Qx
    if !is_terminal(z)
        grad[iu] .= cost.R*u .+ cost.r
    end
    return
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

RD.@autodiff struct ErrorQuadratic{Rot,N,M} <: CostFunction
    model::RobotDynamics.RigidBody{Rot}
    Q::Diagonal{Float64,SVector{12,Float64}}
    R::Diagonal{Float64,SVector{M,Float64}}
    r::SVector{M,Float64}
    c::Float64
    x_ref::SVector{N,Float64}
    q_ind::SVector{4,Int}
    function ErrorQuadratic(model::RobotDynamics.RigidBody{Rot}, 
        Q::Diagonal{<:Real,<:StaticVector{12}}, 
        R::Diagonal{<:Real,<:StaticVector{Nu}},
        r::StaticVector{Nu},
        c::Real,
        x_ref::StaticVector{Nx},
        q_ind::StaticVector{4}
    ) where {Rot,Nx,Nu}
        new{Rot, Nx, Nu}(model, Q, R, r, c, x_ref, q_ind)
    end
end
function Base.copy(c::ErrorQuadratic)
    ErrorQuadratic(c.model, c.Q, c.R, c.r, c.c, c.x_ref, c.q_ind)
end
RD.default_diffmethod(::ErrorQuadratic) = ForwardAD()

"""
    ErrorQuadratic{Rot,N,M}

Cost function of the form:

``\\frac{1}{2} (x_k \\ominus x_d)^T Q_k (x_k \\ominus x_d)``

where ``x_k \\ominus x_d`` is the error state, computed using 
`RobotDynamics.state_diff`. 
This cost function isn't recommended: we've found that `DiagonalQuatCost` usually
    peforms better and is much more computationally efficient.
"""
ErrorQuadratic

state_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = N
control_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = M

function ErrorQuadratic(model::RD.RigidBody{Rot}, Q::Diagonal,
        R::Diagonal,
        x_ref,
        u_ref=(@SVector zeros(eltype(Q),size(R,1))); 
        r=(@SVector zeros(eltype(Q),size(R,1))), 
        c=zero(eltype(Q)),
        q_ind=(@SVector [4,5,6,7])
    ) where {Rot}
    if Rot <: UnitQuaternion && size(Q,1) == size(x_ref,1) 
        Qd = deleteat(Q.diag, 4)
        Q = Diagonal(Qd)
    end
    r += -R*u_ref
    c += 0.5*u_ref'R*u_ref
    return ErrorQuadratic(model, Q, R, r, c, x_ref, q_ind)
end


function RD.evaluate(cost::ErrorQuadratic, x, u)
    dx = RD.state_diff(cost.model, x, cost.x_ref, Rotations.CayleyMap())
    return 0.5*dx'cost.Q*dx + cost.c + 0.5*u'cost.R*u + cost.r'u
end


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