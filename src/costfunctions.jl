import Base: copy, +

#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end

abstract type QuadraticCostFunction{n,m,T} <: CostFunction end

is_diag(cost::QuadraticCostFunction) = is_blockdiag(cost) && cost.Q isa Diagonal && cost.R isa Diagonal
state_dim(::QuadraticCostFunction{n}) where n = n
control_dim(::QuadraticCostFunction{<:Any,m}) where m = m

function (::Type{QC})(Q::AbstractArray, R::AbstractArray;
        H::AbstractArray=SizedMatrix{size(R,1),size(Q,1)}(zeros(eltype(Q), size(R,1), size(Q,1))),
        q::AbstractVector=(@SVector zeros(eltype(Q), size(Q,1))),
        r::AbstractVector=(@SVector zeros(eltype(R), size(R,1))),
        c::Real=zero(eltype(Q)), kwargs...) where QC <: QuadraticCostFunction
    QC(Q, R, H, q, r, c; kwargs...)
end

function stage_cost(cost::QuadraticCostFunction, x::AbstractVector, u::AbstractVector)
    J = 0.5*u'cost.R*u + cost.r'u + stage_cost(cost, x)
    if !is_blockdiag(cost)
        J += u'cost.H*x
    end
    return J
end

function stage_cost(cost::QuadraticCostFunction, x::AbstractVector{T}) where T
    0.5*x'cost.Q*x .+ cost.q'*x .+ cost.c
end

function gradient!(E::QuadraticCostFunction, cost::QuadraticCostFunction, x)
    E.q .= cost.Q*x .+ cost.q
    return false
end

function gradient!(E::QuadraticCostFunction, cost::QuadraticCostFunction, x, u)
    gradient!(E, cost, x)
    E.r .= cost.R*u .+ cost.r
    if !is_blockdiag(cost)
        E.q .+= cost.H'u
        E.r .+= cost.H*x
    end
    return false
end

function hessian!(E::QuadraticCostFunction, cost::QuadraticCostFunction, x)
    if is_diag(cost)
        for i = 1:length(x); E.Q[i,i] = cost.Q[i,i] end
    else
        E.Q .= cost.Q
    end
    return true
end

function hessian!(E::QuadraticCostFunction, cost::QuadraticCostFunction, x, u)
    hessian!(E, cost, x)
    if is_diag(cost)
        for i = 1:length(u); E.R[i,i] = cost.R[i,i]; end
    else
        E.R .= cost.R
    end
    if !is_blockdiag(cost)
        E.H .= cost.H
    end
    return true
end

function Base.copy(c::QC) where QC<:QuadraticCostFunction
    QC(copy(c.Q), copy(c.R), H=copy(c.H), q=copy(c.q), r=copy(c.r), c=c.c,
        terminal=c.terminal, checks=false)
end

# Other methods

function +(c1::QuadraticCostFunction, c2::QuadraticCostFunction)
    @assert state_dim(c1) == state_dim(c2)
    @assert control_dim(c1) == control_dim(c2)
    n,m = state_dim(c1), control_dim(c1)
    H1 = c1 isa DiagonalCost ? zeros(m,n) : c1.H
    H2 = c2 isa DiagonalCost ? zeros(m,n) : c2.H
    QC = promote_type(typeof(c1), typeof(c2))
    QC(c1.Q + c2.Q, c1.R + c2.R, H1 + H2,
       c1.q + c2.q, c1.r + c2.r, c1.c + c2.c,
       checks=false, terminal=c1.terminal && c2.terminal)
end

function invert!(Ginv, cost::QuadraticCostFunction{n,m}) where {n,m}
    ix = 1:n
    iu = n .+ (1:m)
    if is_diag(cost)
        for i = 1:n; Ginv[i,i] = inv(cost.Q[i,i]); end
        if !cost.terminal
            for i = 1:m; Ginv[i+n,i+n] = inv(cost.R[i,i]); end
        end
    elseif is_blockdiag(cost)
        Ginv[ix,ix] .= inv(SizedMatrix{n,n}(cost.Q))
        if !cost.terminal
            Ginv[iu,iu] .= inv(SizedMatrix{m,m}(cost.R))
        end
    else
        G1 = [cost.Q cost.H']
        G2 = [cost.H cost.R ]
        G = [G1; G2]
        Ginv .= inv(G)
    end
end


#######################################################
#              COST FUNCTION INTERFACE                #
#######################################################

"""
    DiagonalCost{n,m,T}

Cost function of the form
```math
\\frac{1}{2} x^T Q x + \\frac{1}{2} u^T R u + q^T x + r^T u + c
```
where ``Q`` and ``R`` are positive semi-definite and positive definite diagonal matrices,
respectively, and ``x`` is `n`-dimensional and ``u`` is `m`-dimensional.

# Constructors
    DiagonalCost(Qd::AbstractVector, Rd::AbstractVector, q::AbstractVector, r::AbstractVector, c; kwargs...)
    DiagonalCost(Q::AbstractMatrix, R::AbstractMatrix, q::AbstractVector, r::AbstractVector, c; kwargs...)
    DiagonalCost(Qd::AbstractVector, Rd::AbstractVector; [q::AbstractVector, r::AbstractVector, c, kwargs...])
    DiagonalCost(Q::AbstractMatrix, R::AbstractMatrix; [q::AbstractVector, r::AbstractVector, c, kwargs...])
"""
struct DiagonalCost{n,m,T} <: QuadraticCostFunction{n,m,T}
    Q::Diagonal{T,SVector{n,T}}
    R::Diagonal{T,SVector{m,T}}
    q::SVector{n,T}
    r::SVector{m,T}
    c::T
    terminal::Bool
    function DiagonalCost(Qd::StaticVector{n}, Rd::StaticVector{m},
                          q::StaticVector{n},  r::StaticVector{m},
                          c::Real; terminal::Bool=false, checks::Bool=true) where {n,m}
        T = promote_type(typeof(c), eltype(Qd), eltype(Rd), eltype(q), eltype(r))
        if checks
            if any(x->x<0, Qd)
                @warn "Q needs to be positive semi-definite."
            elseif any(x->x<=0, Rd) && !terminal
                @warn "R needs to be positive definite."
            end
        end
        new{n,m,T}(Diagonal(SVector(Qd)), Diagonal(SVector(Rd)), SVector(q), SVector(r), T(c), terminal)
    end
end

# strip away the H
@inline function (::Type{<:DiagonalCost})(Q::AbstractArray, R::AbstractArray, H::AbstractArray,
        q::AbstractVector, r::AbstractVector, c; kwargs...)
    DiagonalCost(Q, R, q, r, c; kwargs...)
end

# convert to form for inner constructor
function DiagonalCost(Q::AbstractArray, R::AbstractArray,
        q::AbstractVector, r::AbstractVector, c; kwargs...)
    n,m = length(q), length(r)
    Qd = SVector{n}(diag(Q))
    Rd = SVector{m}(diag(R))
    DiagonalCost(Qd, Rd, SVector{n}(q), SVector{m}(r), c; kwargs...)
end

function DiagonalCost(Q::AbstractVector, R::AbstractVector,
        q::AbstractVector, r::AbstractVector, c::Real; kwargs...)
    DiagonalCost(Diagonal(Q), Diagonal(R), q, r, c; kwargs...)
end

function DiagonalCost(Q::AbstractVector, R::AbstractVector; kwargs...)
    DiagonalCost(Diagonal(Q), Diagonal(R); kwargs...)
end

is_blockdiag(::DiagonalCost) = true
is_diag(::DiagonalCost) = true

function LinearAlgebra.inv(cost::DiagonalCost)
    return DiagonalCost(inv(cost.Q), inv(cost.R), cost.q, cost.r, cost.c, terminal=cost.terminal)
end

function Base.:\(cost::DiagonalCost, z::AbstractKnotPoint)
    x = cost.Q\state(z)
    u = cost.R\control(z)
    return StaticKnotPoint([x;u], z._x, z._u, z.dt, z.t)
end

function change_dimension(cost::DiagonalCost, n::Int, m::Int, ix, iu)
    Qd = zeros(n)
    Rd = zeros(m)
    q = zeros(n)
    r = zeros(m)
    Qd[ix] = diag(cost.Q)
    Rd[iu] = diag(cost.R)
    q[ix] = cost.q
    r[iu] = cost.r
    DiagonalCost(Qd, Rd, q, r, cost.c, terminal=cost.terminal, checks=false)
end

"""
$(TYPEDEF)
Cost function of the form
    1/2xₙᵀ Qf xₙ + qfᵀxₙ +  ∫ ( 1/2xᵀQx + 1/2uᵀRu + xᵀHu + q⁠ᵀx + rᵀu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite

Constructor use any of the following constructors:
```julia
QuadraticCost(Q, R, H, q, r, c)
QuadraticCost(Q, R; H, q, r, c)
QuadraticCost(Q, q, c)
```
Any optional or omitted values will be set to zero(s).
"""
mutable struct QuadraticCost{n,m,T,TQ,TR} <: QuadraticCostFunction{n,m,T}
    Q::TQ                     # Quadratic stage cost for states (n,n)
    R::TR                     # Quadratic stage cost for controls (m,m)
    H::SizedMatrix{m,n,T,2}   # Quadratic Cross-coupling for state and controls (m,n)
    q::MVector{n,T}           # Linear term on states (n,)
    r::MVector{m,T}           # Linear term on controls (m,)
    c::T                      # constant term
    terminal::Bool
    zeroH::Bool
    function (::Type{QC})(Q::TQ, R::TR, H::TH,
            q::Tq, r::Tr, c::Real; checks=true, terminal=false) where {TQ,TR,TH,Tq,Tr,QC<:QuadraticCost}
        @assert size(Q,1) == length(q)
        @assert size(R,1) == length(r)
        @assert size(H) == (length(r), length(q))
        if checks
            if !isposdef(Array(R)) && !terminal
                @warn "R is not positive definite"
            end
            if !ispossemidef(Array(Q))
                @warn "Q is not positive semidefinite"
            end
        end
        zeroH = norm(H,Inf) ≈ 0
        m,n = size(H)
        T = promote_type(eltype(Q), eltype(R), eltype(H), eltype(q), eltype(r), typeof(c))
        new{n,m,T,TQ,TR}(Q,R,H,q,r,c,terminal,zeroH)
    end
    function QuadraticCost{n,m,T,TQ,TR}(qcost::QuadraticCost) where {n,m,T,TQ,TR}
        new{n,m,T,TQ,TR}(qcost.Q, qcost.R, qcost.H, qcost.q, qcost.r, qcost.c, qcost.terminal, qcost.zeroH)
    end
end

state_dim(cost::QuadraticCost) = length(cost.q)
control_dim(cost::QuadraticCost) = length(cost.r)
is_blockdiag(cost::QuadraticCost) = cost.zeroH

function QuadraticCost{T}(n::Int,m::Int; terminal=false) where T
    Q = SizedMatrix{n,n}(Matrix(one(T)*I,n,n))
    R = SizedMatrix{m,m}(Matrix(one(T)*I,m,m))
    H = SizedMatrix{m,n}(zeros(T,m,n))
    q = SizedVector{n}(zeros(T,n))
    r = SizedVector{m}(zeros(T,m))
    c = zero(T)
    QuadraticCost(Q,R,H,q,r,c, checks=false, terminal=terminal)
end

QuadraticCost(cost::QuadraticCost) = cost

function LinearAlgebra.inv(cost::QuadraticCost)
    if norm(cost.H,Inf) ≈ 0
        QuadraticCost(inv(cost.Q), inv(cost.R), cost.H, cost.q, cost.r, cost.c,
            checks=false, terminal=cost.terminal)
    else
        m,n = size(cost.H)
        H1 = [cost.Q cost.H']
        H2 = [cost.H cost.R ]
        H = [H1; H2]
        Hinv = inv(H)
        ix = 1:n
        iu = n .+ (1:m)
        Q = SizedMatrix{n,n}(Hinv[ix, ix])
        R = SizedMatrix{m,m}(Hinv[iu, iu])
        H = SizedMatrix{m,n}(Hinv[iu, ix])
        QuadraticCost(Q,R,H, cost.q, cost.r, cost.c,
            checks=false, terminal=cost.terminal)
    end
end

@inline function (::Type{<:QuadraticCost})(dcost::DiagonalCost)
    QuadraticCost(dcost,Q, dcost.R, q=dcost.q, r=dcost.r, c=dcost.c, terminal=dcost.terminal)
end

function Base.convert(::Type{<:QuadraticCost}, dcost::DiagonalCost)
    QuadraticCost(dcost.Q, dcost.R, q=dcost.q, r=dcost.r, c=dcost.c, terminal=dcost.terminal)
end

Base.promote_rule(::Type{<:QuadraticCostFunction}, ::Type{<:QuadraticCostFunction}) = QuadraticCost

function Base.promote_rule(::Type{<:QuadraticCost{n,m,T1,Q1,R1}},
                           ::Type{<:QuadraticCost{n,m,T2,Q2,R2}}) where {n,m,T1,T2,Q1,Q2,R1,R2}
    T = promote_type(T1,T2)
    function diag_type(T1,T2,n)
        elT = promote_type(eltype(T1), eltype(T2))
        if T1 == T2
            return T1
        elseif (T1 <: Diagonal) && (T2 <: Diagonal)
            return Diagonal{elT,MVector{n,elT}}
        else
            return SizedMatrix{n,n,elT,2}
        end
    end
    Q = diag_type(Q1,Q2,n)
    R = diag_type(R1,R2,m)
    QuadraticCost{n,m,T, Q, R}
end

@inline Base.convert(::Type{QC}, cost::QuadraticCost) where QC <: QuadraticCost = QC(cost)

"""
$(SIGNATURES)
Cost function of the form
``(x-x_f)^T Q (x_x_f) + u^T R u``
R must be positive definite, Q must be positive semidefinite
"""
function LQRCost(Q::AbstractArray, R::AbstractArray,
        xf::AbstractVector, uf=(@SVector zeros(size(R,1))); kwargs...)
    H = @SMatrix zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    return DiagonalCost(Q, R, H, q, r, c, kwargs...)
end

function LQRCost(Q::Diagonal{<:Any,<:SVector{n}},R::Diagonal{<:Any,<:SVector{m}},
        xf::AbstractVector, uf=(@SVector zeros(m)); kwargs...) where {n,m}
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    return DiagonalCost(Q, R, q, r, c; kwargs...)
end



############################################################################################
#                        QUADRATIC QUATERNION COST FUNCTION
############################################################################################

struct QuadraticQuatCost{T,N,M,N4} <: CostFunction
    Q::Diagonal{T,SVector{N,T}}
    R::Diagonal{T,SVector{M,T}}
    q::SVector{N,T}
    r::SVector{M,T}
    c::T
    w::T
    q_ref::SVector{4,T}
    q_ind::SVector{4,Int}
    Iq::SMatrix{N,4,T,N4}
    function QuadraticQuatCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
            q::SVector{N,T}, r::SVector{M,T}, c::T, w::T,
            q_ref::SVector{4,T}, q_ind::SVector{4,Int}) where {T,N,M}
        Iq = @MMatrix zeros(N,4)
        for i = 1:4
            Iq[q_ind[i],i] = 1
        end
        Iq = SMatrix{N,4}(Iq)
        return new{T,N,M,N*4}(Q, R, q, r, c, w, q_ref, q_ind, Iq)
    end
end


state_dim(::QuadraticQuatCost{T,N,M}) where {T,N,M} = N
control_dim(::QuadraticQuatCost{T,N,M}) where {T,N,M} = M

function QuadraticQuatCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}};
        q=(@SVector zeros(N)), r=(@SVector zeros(M)), c=zero(T), w=one(T),
        q_ref=(@SVector [1.0,0,0,0]), q_ind=(@SVector [4,5,6,7])) where {T,N,M}
    QuadraticQuatCost(Q, R, q, r, c, q_ref, q_ind)
end

function stage_cost(cost::QuadraticQuatCost, x::SVector, u::SVector)
    stage_cost(cost, x) + 0.5*u'cost.R*u + cost.r'u
end

function stage_cost(cost::QuadraticQuatCost, x::SVector)
    J = 0.5*x'cost.Q*x + cost.q'x + cost.c
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    J += cost.w*min(1+dq, 1-dq)
end

function gradient(cost::QuadraticQuatCost{T,N,M}, x::SVector, u::SVector) where {T,N,M}
    Qx = cost.Q*x + cost.q
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    if dq < 0
        Qx += cost.w*cost.Iq*cost.q_ref
    else
        Qx -= cost.w*cost.Iq*cost.q_ref
    end
    Qu = cost.R*u + cost.r
    return Qx, Qu
end

function hessian(cost::QuadraticQuatCost, x::SVector{N}, u::SVector{M}) where {N,M}
    Qxx = cost.Q
    Quu = cost.R
    Qux = @SMatrix zeros(M,N)
    return Qxx, Quu, Qux
end

function QuatLQRCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}}, xf,
        uf=(@SVector zeros(M)); w=one(T), quat_ind=(@SVector [4,5,6,7])) where {T,N,M}
    r = -R*uf
    q = -Q*xf
    c = 0.5*xf'Q*xf + 0.5*uf'R*uf
    q_ref = xf[quat_ind]
    return QuadraticQuatCost(Q, R, q, r, c, w, q_ref, quat_ind)
end

function change_dimension(cost::QuadraticQuatCost, n, m)
    n0,m0 = state_dim(cost), control_dim(cost)
    Q_diag = diag(cost.Q)
    R_diag = diag(cost.R)
    q = cost.q
    r = cost.r
    if n0 != n
        dn = n - n0  # assumes n > n0
        pad = @SVector zeros(dn)
        Q_diag = [Q_diag; pad]
        q = [q; pad]
    end
    if m0 != m
        dm = m - m0  # assumes m > m0
        pad = @SVector zeros(dm)
        R_diag = [R_diag; pad]
        r = [r; pad]
    end
    QuadraticQuatCost(Diagonal(Q_diag), Diagonal(R_diag), q, r, cost.c, cost.w,
        cost.q_ref, cost.q_ind)
end

function (+)(cost1::QuadraticQuatCost, cost2::QuadraticCost)
    @assert state_dim(cost1) == state_dim(cost2)
    @assert control_dim(cost1) == control_dim(cost2)
    @assert norm(cost2.H) ≈ 0
    QuadraticQuatCost(cost1.Q + cost2.Q, cost1.R + cost2.R,
        cost1.q + cost2.q, cost1.r + cost2.r, cost1.c + cost2.c,
        cost1.w, cost1.q_ref, cost1.q_ind)
end

(+)(cost1::QuadraticCost, cost2::QuadraticQuatCost) = cost2 + cost1


#
#
# struct ErrorQuadratic{Rot,N,M} <: CostFunction
#     model::RigidBody{Rot}
#     Q::Diagonal{Float64,SVector{12,Float64}}
#     R::Diagonal{Float64,SVector{M,Float64}}
#     r::SVector{M,Float64}
#     c::Float64
#     x_ref::SVector{N,Float64}
#     q_ind::SVector{4,Int}
# end
#
#
# state_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = N
# control_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = M
#
# function ErrorQuadratic(model::RigidBody{Rot}, Q::Diagonal{T,<:SVector{12}},
#         R::Diagonal{T,<:SVector{M}},
#         x_ref::SVector{N}, u_ref=(@SVector zeros(T,M)); r=(@SVector zeros(T,M)), c=zero(T),
#         q_ind=(@SVector [4,5,6,7])) where {T,N,M,Rot}
#     r += -R*u_ref
#     c += 0.5*u_ref'R*u_ref
#     return ErrorQuadratic{Rot,N,M}(model, Q, R, r, c, x_ref, q_ind)
# end
#
# function stage_cost(cost::ErrorQuadratic, x::SVector)
#     dx = state_diff(cost.model, x, cost.x_ref)
#     return 0.5*dx'cost.Q*dx + cost.c
# end
#
# function stage_cost(cost::ErrorQuadratic, x::SVector, u::SVector)
#     stage_cost(cost, x) + 0.5*u'cost.R*u + cost.r'u
# end
#
# function cost_expansion(cost::ErrorQuadratic{Rot}, model::AbstractModel,
#         z::KnotPoint{T,N,M}, G) where {T,N,M,Rot<:UnitQuaternion}
#     x,u = state(z), control(z)
#     model = cost.model
#     Q = cost.Q
#     q = orientation(model, x)
#     q_ref = orientation(model, cost.x_ref)
#     dq = SVector(q_ref\q)
#     err = state_diff(model, x, cost.x_ref)
#     dx = @SVector [err[1],  err[2],  err[3],
#                     dq[1],   dq[2],   dq[3],   dq[4],
#                    err[7],  err[8],  err[9],
#                    err[10], err[11], err[12]]
#     G = state_diff_jacobian(model, dx) # n × dn
#
#     # Gradient
#     dmap = inverse_map_jacobian(model, dx) # dn × n
#     Qx = G'dmap'Q*err
#     Qu = cost.R*u
#
#     # Hessian
#     ∇jac = inverse_map_∇jacobian(model, dx, Q*err)
#     Qxx = G'dmap'Q*dmap*G + G'∇jac*G + ∇²differential(model, x, dmap'Q*err)
#     Quu = cost.R
#     Qux = @SMatrix zeros(M,N-1)
#     return Qxx, Quu, Qux, Qx, Qu
# end
#
# function cost_expansion(cost::ErrorQuadratic, model::AbstractModel,
#         z::KnotPoint{T,N,M}, G) where {T,N,M}
#     x,u = state(z), control(z)
#     model = cost.model
#     q = orientation(model, x)
#     q_ref = orientation(model, cost.x_ref)
#     err = state_diff(model, x, cost.x_ref)
#     dx = err
#     G = state_diff_jacobian(model, dx) # n × n
#
#     # Gradient
#     dmap = inverse_map_jacobian(model, dx) # n × n
#     Qx = G'dmap'cost.Q*err
#     Qu = cost.R*u + cost.r
#
#     # Hessian
#     Qxx = G'dmap'cost.Q*dmap*G
#     Quu = cost.R
#     Qux = @SMatrix zeros(M,N)
#     return Qxx, Quu, Qux, Qx, Qu
# end
#
# function change_dimension(cost::ErrorQuadratic, n, m)
#     n0,m0 = state_dim(cost), control_dim(cost)
#     Q_diag = diag(cost.Q)
#     R_diag = diag(cost.R)
#     r = cost.r
#     if n0 != n
#         dn = n - n0  # assumes n > n0
#         pad = @SVector zeros(dn) # assume the new states don't have quaternions
#         Q_diag = [Q_diag; pad]
#     end
#     if m0 != m
#         dm = m - m0  # assumes m > m0
#         pad = @SVector zeros(dm)
#         R_diag = [R_diag; pad]
#         r = [r; pad]
#     end
#     ErrorQuadratic(cost.model, Diagonal(Q_diag), Diagonal(R_diag), r, cost.c,
#         cost.x_ref, cost.q_ind)
# end
#
# function (+)(cost1::ErrorQuadratic, cost2::QuadraticCost)
#     @assert control_dim(cost1) == control_dim(cost2)
#     @assert norm(cost2.H) ≈ 0
#     @assert norm(cost2.q) ≈ 0
#     if state_dim(cost2) == 13
#         rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
#         Q2 = Diagonal(diag(cost2.Q)[rm_quat])
#     else
#         Q2 = cost2.Q
#     end
#     ErrorQuadratic(cost1.model, cost1.Q + Q2, cost1.R + cost2.R,
#         cost1.r + cost2.r, cost1.c + cost2.c,
#         cost1.x_ref, cost1.q_ind)
# end
#
# (+)(cost1::QuadraticCost, cost2::ErrorQuadratic) = cost2 + cost1
#
#
#
#
