import Base.copy

#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end
CostTrajectory = Vector{C} where C <: CostFunction

"Calculate unconstrained cost for X and U trajectories"
function cost(c::CostTrajectory, X::VectorTrajectory{T}, U::VectorTrajectory{T},dt::Vector{T})::T where T
    N = length(X)
    J = 0.0
    for k = 1:N-1
        J += stage_cost(c[k],X[k],U[k],dt[k])
    end
    J += stage_cost(c[N],X[N])
    return J
end

"$(TYPEDEF) Expansion of cost function"
struct Expansion{T<:AbstractFloat,Q<:AbstractMatrix,R<:AbstractMatrix}
    x::Vector{T}
    u::Vector{T}
    xx::Q
    uu::R
    ux::Matrix{T}
end

function Expansion{T,Q,R}(n::Int, m::Int) where {T,Q,R}
    x = zeros(T,n)
    u = zeros(T,m)
    xx = Q(zeros(T,n,n))
    uu = R(zeros(T,m,m))
    ux = zeros(T,m,n)
    Expansion(x, u, xx, uu, ux)
end
Expansion{T}(n::Int, m::Int) where T = Expansion{T,Matrix{T},Matrix{T}}(n,m)

import Base./, Base.*
function *(e::Expansion, a::Real)
    e.x .*= a
    e.u .*= a
    e.xx .*= a
    e.uu .*= a
    e.ux .*= a
    return nothing
end
*(a::Real, e::Expansion) = e*a

function /(e::Expansion,a::Real)
    e.x ./= a
    e.u ./= a
    e.xx ./= a
    e.uu ./= a
    e.ux ./= a
    return nothing
end

function copy(e::Expansion)
    Expansion(copy(e.x),copy(e.u),copy(e.xx),copy(e.uu),copy(e.ux))
end

function reset!(e::Expansion)
    !isempty(e.x) ? e.x .= zero(e.x) : nothing
    !isempty(e.u) ? e.u .= zero(e.u) : nothing
    !isempty(e.xx) ? e.xx .= zero(e.xx) : nothing
    !isempty(e.uu) ? e.uu .= zero(e.uu) : nothing
    !isempty(e.ux) ? e.ux .= zero(e.ux) : nothing
    return nothing
end

ExpansionTrajectory{T} = Vector{Expansion{T}} where T <: AbstractFloat

function reset!(et::ExpansionTrajectory)
    for e in et
        reset!(e)
    end
end

"""
$(TYPEDEF)
Cost function of the form
    1/2xₙᵀ Qf xₙ + qfᵀxₙ +  ∫ ( 1/2xᵀQx + 1/2uᵀRu + xᵀHu + q⁠ᵀx  rᵀu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
mutable struct QuadraticCost{T} <: CostFunction
    Q::AbstractMatrix{T}                 # Quadratic stage cost for states (n,n)
    R::AbstractMatrix{T}                 # Quadratic stage cost for controls (m,m)
    H::AbstractMatrix{T}                 # Quadratic Cross-coupling for state and controls (n,m)
    q::AbstractVector{T}                 # Linear term on states (n,)
    r::AbstractVector{T}                 # Linear term on controls (m,)
    c::T                                 # constant term
    function QuadraticCost(Q::AbstractMatrix{T}, R::AbstractMatrix{T}, H::AbstractMatrix{T},
            q::AbstractVector{T}, r::AbstractVector{T}, c::T) where T
        if !isposdef(R)
            @warn "R is not positive definite"
        end
        if !ispossemidef(Q)
            err = ArgumentError("Q must be positive semi-definite")
            throw(err)
        end

        new{T}(Q,R,H,q,r,c)
    end
end

get_sizes(cost::QuadraticCost) = (size(cost.Q,1),size(cost.R,1))

function QuadraticCost(Q,R; H=zeros(size(R,1), size(Q,1)), q=zeros(size(Q,1)),
        r=zeros(size(R,1)), c=0.0)
    QuadraticCost(Q,R,H,q,r,c)
end

function QuadraticCost(Q,q,c)
    QuadraticCost(Q,zeros(0,0),zeros(0,size(Q,1)),q,zeros(0),c)
end


"""
$(SIGNATURES)
Cost function of the form
    1/2(xₙ-x_f)ᵀ Qf (xₙ - x_f) + 1/2 ∫ ( (x-x_f)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
function LQRCost(Q::AbstractArray, R::AbstractArray, xf::AbstractVector)
    H = zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = zeros(size(R,1))
    c = 0.5*xf'*Q*xf
    return QuadraticCost(Q, R, H, q, r, c)
end

function LQRCostTerminal(Qf::AbstractArray,xf::AbstractVector)
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf
    return QuadraticCost(Qf,zeros(0,0),zeros(0,size(Qf,1)),qf,zeros(0),cf)
end

function stage_cost(cost::QuadraticCost, x::AbstractVector, u::AbstractVector)
    0.5*x'cost.Q*x + 0.5*u'*cost.R*u + cost.q'x + cost.r'u + cost.c + u'*cost.H*x
end

function stage_cost(cost::QuadraticCost, x::AbstractVector, u::AbstractVector, dt)
    (0.5*x'cost.Q*x + 0.5*u'*cost.R*u + cost.q'x + cost.r'u + cost.c + u'*cost.H*x)*dt
end

function stage_cost(cost::QuadraticCost, xN::AbstractVector{T}) where T
    0.5*xN'cost.Q*xN + cost.q'*xN + cost.c
end

function cost_expansion!(Q::Expansion{T}, cost::QuadraticCost,
        x::AbstractVector{T}, u::AbstractVector{T}, dt::T) where T
    Q.x .= cost.Q*x + cost.q + cost.H'*u
    Q.u .= cost.R*u + cost.r + cost.H*x
    Q.xx .= cost.Q
    Q.uu .= cost.R
    Q.ux .= cost.H
    Q*dt
    return nothing
end

function cost_expansion!(S::Expansion{T}, cost::QuadraticCost, xN::AbstractVector{T}) where T
    S.xx .= cost.Q
    S.x .= cost.Q*xN + cost.q
    return nothing
end

function gradient!(grad, cost::QuadraticCost,
        x::AbstractVector, u::AbstractVector)
    grad.x .= (cost.Q*x + cost.q + cost.H'*u)
    grad.u .= (cost.R*u + cost.r + cost.H*x)
    return nothing
end

function gradient!(grad, cost::QuadraticCost, xN::AbstractVector)
    grad .= cost.Q*xN + cost.q
    return nothing
end

function hessian!(hess, cost::QuadraticCost,
        x::AbstractVector, u::AbstractVector)
    hess.xx .= cost.Q
    hess.uu .= cost.R
    hess.ux .= cost.H
    hess.xu .= cost.H'
    return nothing
end

function hessian!(hess, cost::QuadraticCost, xN::AbstractVector)
    hess .= cost.Q
    return nothing
end

function copy(cost::QuadraticCost)
    return QuadraticCost(copy(cost.Q), copy(cost.R), copy(cost.H), copy(cost.q), copy(cost.r), copy(cost.c))
end

"""
$(TYPEDEF)
Cost function of the form
    ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf
"""
struct GenericCost <: CostFunction
    ℓ::Function             # Stage cost
    ℓf::Function            # Terminal cost
    expansion::Function     # 2nd order Taylor Series Expansion of the form,  Q,R,H,q,r = expansion(x,u)
    n::Int                  #                                                     Qf,qf = expansion(xN)
    m::Int
end

"""
$(SIGNATURES)
Create a Generic Cost, specifying the gradient and hessian of the cost function analytically

# Arguments
* hess: multiple-dispatch function of the form,
    Q,R,H = hess(x,u) with sizes (n,n), (m,m), (m,n)
    Qf = hess(xN) with size (n,n)
* grad: multiple-dispatch function of the form,
    q,r = grad(x,u) with sizes (n,), (m,)
    qf = grad(x,u) with size (n,)

"""
function GenericCost(ℓ::Function, ℓf::Function, grad::Function, hess::Function, n::Int, m::Int)
    function expansion(x::Vector{T},u::Vector{T}) where T
        Q,R,H = hess(x,u)
        q,r = grad(x,u)
        return Q,R,H,q,r
    end
    expansion(xN) = hess(xN), grad(xN)
    GenericCost(ℓ,ℓf, expansion, n,m)
end


"""
$(SIGNATURES)
Create a Generic Cost. Gradient and Hessian information will be determined using ForwardDiff

# Arguments
* ℓ: stage cost function of the form J = ℓ(x,u)
* ℓf: terminal cost function of the form J = ℓ(xN)
"""
function GenericCost(ℓ::Function, ℓf::Function, n::Int, m::Int)
    linds = LinearIndices(zeros(n+m,n+m))
    xinds = 1:n
    uinds = n .+(1:m)
    inds = (x=xinds, u=uinds, xx=linds[xinds,xinds], uu=linds[uinds,uinds], ux=linds[uinds,xinds])
    expansion = auto_expansion_function(ℓ,ℓf,n,m)

    GenericCost(ℓ,ℓf, expansion, n,m)
end

function auto_expansion_function(ℓ::Function,ℓf::Function,n::Int,m::Int)
    z = zeros(n+m)
    hess = zeros(n+m,n+m)
    grad = zeros(n+m)
    qf,Qf = zeros(n), zeros(n,n)

    linds = LinearIndices(hess)
    xinds = 1:n
    uinds = n .+(1:m)
    inds = (x=xinds, u=uinds, xx=linds[xinds,xinds], uu=linds[uinds,uinds], ux=linds[uinds,xinds])
    function ℓ_aug(z::Vector{T}) where T
        x = view(z,xinds)
        u = view(z,uinds)
        ℓ(x,u)
    end
    function expansion(x::Vector{T},u::Vector{T}) where T
        z[inds.x] = x
        z[inds.u] = u
        ForwardDiff.hessian!(hess, ℓ_aug, z)
        Q = view(hess,inds.xx)
        R = view(hess,inds.uu)
        H = view(hess,inds.ux)

        ForwardDiff.gradient!(grad, ℓ_aug, z)
        q = view(grad,inds.x)
        r = view(grad,inds.u)
        return Q,R,H,q,r
    end
    function expansion(xN::Vector{T}) where T
        ForwardDiff.gradient!(qf,ℓf,xN)
        ForwardDiff.hessian!(Qf,ℓf,xN)
        return Qf, qf
    end
end

stage_cost(cost::GenericCost, x::Vector{T}, u::Vector{T}) where T = cost.ℓ(x,u)
stage_cost(cost::GenericCost, xN::Vector{T}) where T = cost.ℓf(xN)

function cost_expansion!(Q::Expansion{T}, cost::GenericCost, x::Vector{T},
        u::Vector{T}) where T

    e = cost.expansion(x,u)

    Q.x .= e[4]
    Q.u .= e[5]
    Q.xx .= e[1]
    Q.uu .= e[2]
    Q.ux .= e[3]
    return nothing
end

function cost_expansion!(S::Expansion{T}, cost::GenericCost, xN::Vector{T}) where T
    Qf, qf = cost.expansion(xN)
    S.xx .= Qf
    S.x .= qf
    return nothing
end

copy(cost::GenericCost) = GenericCost(cost.ℓ,cost.ℓ,cost.n,cost.m)
