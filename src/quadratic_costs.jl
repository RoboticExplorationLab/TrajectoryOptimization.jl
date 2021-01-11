"""
An abstract type that represents any [`CostFunction`](@ref) of the form
```math
\\frac{1}{2} x^T Q x + \\frac{1}{2} u^T R u + u^T H x + q^T x + r^T u + c
```

These types all support the following methods
* [`is_diag`](@ref)
* [`is_blockdiag`](@ref)
* [`invert!`](@ref)

As well as standard addition.
"""
abstract type QuadraticCostFunction{n,m,T} <: CostFunction end

"""
    is_diag(::QuadraticCostFunction)

Determines if the hessian of a quadratic cost function is strictly diagonal.
"""
is_diag(cost::QuadraticCostFunction) = is_blockdiag(cost) && cost.Q isa Diagonal && cost.R isa Diagonal

"""
    is_diag(::QuadraticCostFunction)

Determines if the hessian of a quadratic cost function is block diagonal (i.e. ``\\norm(H) = 0``).
"""
is_blockdiag(::QuadraticCostFunction) = false
state_dim(::QuadraticCostFunction{n}) where n = n
control_dim(::QuadraticCostFunction{<:Any,m}) where m = m

function (::Type{QC})(Q::AbstractArray, R::AbstractArray;
        H::AbstractArray=SizedMatrix{size(R,1),size(Q,1)}(zeros(eltype(Q), size(R,1), size(Q,1))),
        q::AbstractVector=(@SVector zeros(eltype(Q), size(Q,1))),
        r::AbstractVector=(@SVector zeros(eltype(R), size(R,1))),
        c::Real=zero(eltype(Q)), kwargs...) where QC <: QuadraticCostFunction
    QC(Q, R, H, q, r, c; kwargs...)
end

function QuadraticCostFunction(Q::AbstractArray, R::AbstractArray,
        H::AbstractArray, q::AbstractVector, r::AbstractVector, c::Real;
        kwargs...)
    if LinearAlgebra.isdiag(Q) && LinearAlgebra.isdiag(R) && norm(H) ≈ 0
        DiagonalCost(diag(Q), diag(R), q, r, c)
    else
        QuadraticCost(Q, R, H, q, r, c; kwargs...)
    end
end

"""
    stage_cost(costfun::CostFunction, x, u)
    stage_cost(costfun::CostFunction, x)

Calculate the scalar cost using `costfun` given state `x` and control `u`. If only the
state is provided, it is assumed it is a terminal cost.
"""
function stage_cost(cost::QuadraticCostFunction, x::AbstractVector, u::AbstractVector)
    J = 0.5*u'cost.R*u + dot(cost.r,u) + stage_cost(cost, x)
    if !is_blockdiag(cost)
        J += u'cost.H*x
    end
    return J
end

function stage_cost(cost::QuadraticCostFunction, x::AbstractVector{T}) where T
    0.5*x'cost.Q*x + dot(cost.q,x) + cost.c
end

"""
    gradient!(E::QuadraticCostFunction, costfun::CostFunction, z::AbstractKnotPoint, [cache])

Evaluate the gradient of the cost function `costfun` at state `x` and control `u`, storing
    the result in `E.q` and `E.r`. Return a `true` if the gradient is constant, and `false`
    otherwise.

If `is_terminal(z)` is true, it will only calculate the gradientwith respect to the terminal
state.

The optional `cache` argument provides an optional method to pass in extra memory to
facilitate computation of cost expansion. It is vector of length 4, with the following
entries: `[grad, hess, grad_term, hess_term]`, where `grad` and `hess` are the caches 
for gradients and Hessians repectively, and the `[]_term` entries are the caches for the
terminal cost function.
"""
function gradient!(E, cost::QuadraticCostFunction, z::AbstractKnotPoint, 
        cache=ExpansionCache(cost))
    x,u = state(z), control(z)
    E.q .= cost.Q*x .+ cost.q
    if !is_terminal(z)
        E.r .= cost.R*u .+ cost.r
        if !is_blockdiag(cost)
            E.q .+= cost.H'u
            E.r .+= cost.H*x
        end
    end
    return false
end

"""
    hessian!(E, costfun::CostFunction, z::AbstractKnotPoint, [cache])

Evaluate the hessian of the cost function `costfun` at knotpoint `z`.
    the result in `E.Q`, `E.R`, and `E.H`. Return a `true` if the hessian is constant, and `false`
    otherwise.

If `is_terminal(z)` is true, it will only calculate the Hessian with respect to the terminal
state.

The optional `cache` argument provides an optional method to pass in extra memory to
facilitate computation of cost expansion. It is vector of length 4, with the following
entries: `[grad, hess, grad_term, hess_term]`, where `grad` and `hess` are the caches 
for gradients and Hessians repectively, and the `[]_term` entries are the caches for the
terminal cost function.
"""
function hessian!(E, cost::QuadraticCostFunction, z::AbstractKnotPoint,
        cache=ExpansionCache(cost))
    x,u = state(z), control(z)
    if is_diag(cost)
        for i = 1:length(x); E.Q[i,i] = cost.Q[i,i] end
    else
        E.Q .= cost.Q
    end
    if !is_terminal(z)
        if is_diag(cost)
            for i = 1:length(u); E.R[i,i] = cost.R[i,i]; end
        else
            E.R .= cost.R
        end
        if !is_blockdiag(cost)
            E.H .= cost.H
        end
    end
    return true
end

function Base.copy(c::QC) where QC<:QuadraticCostFunction
    QC(copy(c.Q), copy(c.R), H=copy(c.H), q=copy(c.q), r=copy(c.r), c=c.c,
        terminal=c.terminal, checks=false)
end

# Other methods
"""
    set_LQR_goal!(cost::QuadraticCostFunction, xf)
    set_LQR_goal!(cost::QuadraticCostFunction, xf, uf)

Change the reference state and control for an LQR tracking cost.
Only changes `q` and `r`, and not the constant term `c`. If `uf` is 
not passed in, it isn't changed. 
"""
function set_LQR_goal!(cost::QuadraticCostFunction, xf)
    cost.q .= -cost.Q * xf
    return nothing
end

function set_LQR_goal!(cost::QuadraticCostFunction, xf, uf)
    set_LQR_goal!(cost, xf)
    cost.r .= -cost.R * uf
    return nothing
end

function +(c1::QuadraticCostFunction, c2::QuadraticCostFunction)
    @assert state_dim(c1) == state_dim(c2)
    @assert control_dim(c1) == control_dim(c2)
    n,m = state_dim(c1), control_dim(c1)
    H1 = is_diag(c1) ? zeros(m,n) : c1.H
    H2 = is_diag(c2) ? zeros(m,n) : c2.H
    QC = promote_type(typeof(c1), typeof(c2))
    QC(c1.Q + c2.Q, c1.R + c2.R, H1 + H2,
       c1.q + c2.q, c1.r + c2.r, c1.c + c2.c,
       checks=false, terminal=c1.terminal && c2.terminal)
end

"""
    invert!(Ginv, cost::QuadraticCostFunction)

Invert the hessian of the cost function, storing the result in `Ginv`. Performs the inversion
    efficiently, depending on the structure of the Hessian (diagonal or block diagonal).
"""
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
    DiagonalCost(Qd, Rd, q, r, c; kwargs...)
    DiagonalCost(Q, R, q, r, c; kwargs...)
    DiagonalCost(Qd, Rd; [q, r, c, kwargs...])
    DiagonalCost(Q, R; [q, r, c, kwargs...])

where `Qd` and `Rd` are the diagonal vectors, and `Q` and `R` are matrices.

Any optional or omitted values will be set to zero(s). The keyword arguments are
* `terminal` - A `Bool` specifying if the cost function is terminal cost or not.
* `checks` - A `Bool` specifying if `Q` and `R` will be checked for the required definiteness.
"""
struct DiagonalCost{n,m,T} <: QuadraticCostFunction{n,m,T}
    Q::Diagonal{T,SVector{n,T}}
    R::Diagonal{T,SVector{m,T}}
    q::MVector{n,T}
    r::MVector{m,T}
    c::T
    terminal::Bool
    function DiagonalCost(Qd::StaticVector{n}, Rd::StaticVector{m},
                          q::StaticVector{n},  r::StaticVector{m},
                          c::Real; terminal::Bool=false, checks::Bool=true, kwargs...) where {n,m}
        T = promote_type(typeof(c), eltype(Qd), eltype(Rd), eltype(q), eltype(r))
        if checks
            if any(x->x<0, Qd)
                @warn "Q needs to be positive semi-definite."
            elseif any(x->x<=0, Rd) && !terminal
                @warn "R needs to be positive definite."
            end
        end
        new{n,m,T}(Diagonal(SVector(Qd)), Diagonal(SVector(Rd)), q, r, T(c), terminal)
    end
end

Base.copy(c::DiagonalCost) = 
    DiagonalCost(c.Q.diag, c.R.diag, copy(c.q), copy(c.r), c.c, 
        checks=false, terminal=c.terminal)

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
    DiagonalCost(Qd, Rd, convert(MVector{n},q), convert(MVector{m},r), c; kwargs...)
end


# Pass in vectors for Q and R
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
    QuadraticCost{n,m,T,TQ,TR}

Cost function of the form
```math
\\frac{1}{2} x^T Q x + \\frac{1}{2} u^T R u + u^T H x + q^T x + r^T u + c
```
where ``R`` must be positive definite, ``Q`` and ``Q_f`` must be positive semidefinite.

The type parameters `TQ` and `TR` specify the type of ``Q`` and ``R``.

# Constructor
    QuadraticCost(Q, R, H, q, r, c; kwargs...)
    QuadraticCost(Q, R; H, q, r, c, kwargs...)

Any optional or omitted values will be set to zero(s). The keyword arguments are
* `terminal` - A `Bool` specifying if the cost function is terminal cost or not.
* `checks` - A `Bool` specifying if `Q` and `R` will be checked for the required definiteness.
"""
mutable struct QuadraticCost{n,m,T,TQ,TR} <: QuadraticCostFunction{n,m,T}
    Q::TQ                     # Quadratic stage cost for states (n,n)
    R::TR                     # Quadratic stage cost for controls (m,m)
    H::SizedMatrix{m,n,T,2,Matrix{T}}   # Quadratic Cross-coupling for state and controls (m,n)
    q::MVector{n,T}           # Linear term on states (n,)
    r::MVector{m,T}           # Linear term on controls (m,)
    c::T                      # constant term
    terminal::Bool
    zeroH::Bool
    function (::Type{QC})(Q::TQ, R::TR, H::TH,
            q::Tq, r::Tr, c::Real; checks=true, terminal=false, kwargs...) where {TQ,TR,TH,Tq,Tr,QC<:QuadraticCost}
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
    LQRCost(Q, R, xf, [uf; kwargs...])

Convenience constructor for a `QuadraticCostFunction` of the form:
```math
\\frac{1}{2} (x-x_f)^T Q (x-xf) + \\frac{1}{2} (u-u_f)^T R (u-u_f)
```

If ``Q`` and ``R`` are diagonal, the output will be a `DiagonalCost`, otherwise it will
be a `QuadraticCost`.
"""
function LQRCost(Q::AbstractArray, R::AbstractArray,
        xf::AbstractVector, uf=(@SVector zeros(size(R,1))); kwargs...)
    H = @SMatrix zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    return QuadraticCostFunction(Q, R, H, q, r, c, kwargs...)
end

function LQRCost(Q::Diagonal{<:Any,<:SVector{n}},R::Diagonal{<:Any,<:SVector{m}},
        xf::AbstractVector, uf=(@SVector zeros(m)); kwargs...) where {n,m}
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    return DiagonalCost(Q, R, q, r, c; kwargs...)
end