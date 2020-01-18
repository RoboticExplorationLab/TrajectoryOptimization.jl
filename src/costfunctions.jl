import Base.copy

export
    CostFunction



#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end


#######################################################
#              COST FUNCTION INTERFACE                #
#######################################################

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
mutable struct QuadraticCost{TQ,TR,TH,Tq,Tr,T} <: CostFunction
    Q::TQ                 # Quadratic stage cost for states (n,n)
    R::TR                 # Quadratic stage cost for controls (m,m)
    H::TH                 # Quadratic Cross-coupling for state and controls (n,m)
    q::Tq                 # Linear term on states (n,)
    r::Tr                 # Linear term on controls (m,)
    c::T                                 # constant term
    function QuadraticCost(Q::TQ, R::TR, H::TH,
            q::Tq, r::Tr, c::T; checks=true) where {TQ,TR,TH,Tq,Tr,T}
        @assert size(Q,1) == length(q)
        @assert size(R,1) == length(r)
        @assert size(H) == (length(r), length(q))
        if checks
            if !isposdef(Array(R))
                @warn "R is not positive definite"
            end
            if !ispossemidef(Array(Q))
                err = ArgumentError("Q must be positive semi-definite")
                throw(err)
            end
        end
        new{TQ,TR,TH,Tq,Tr,T}(Q,R,H,q,r,c)
    end
end

state_dim(cost::QuadraticCost) = length(cost.q)
control_dim(cost::QuadraticCost) = length(cost.r)

# Constructors
function QuadraticCost(Q,R; H=similar(Q,size(R,1), size(Q,1)), q=zeros(size(Q,1)),
        r=zeros(size(R,1)), c=0.0, checks=true)
    QuadraticCost(Q,R,H,q,r,c, checks=checks)
end

function QuadraticCost(Q::Union{Diagonal{T,SVector{N,T}}, SMatrix{N,N,T}},
            R::Union{Diagonal{T,SVector{M,T}}, SMatrix{M,M,T}};
            H=(@SMatrix zeros(T,M,N)),
            q=(@SVector zeros(T,N)),
            r=(@SVector zeros(T,M)),
            c=0.0, checks=true) where {T,N,M}
    QuadraticCost(Q,R,H,q,r,c, checks=checks)
end

function QuadraticCost(Q,q,c; checks=true)
    QuadraticCost(Q,zeros(0,0),zeros(0,size(Q,1)),q,zeros(0),c,checks=checks)
end

"""
$(SIGNATURES)
Cost function of the form
``(x-x_f)^T Q (x_x_f) + u^T R u``
R must be positive definite, Q must be positive semidefinite
"""
function LQRCost(Q::AbstractArray, R::AbstractArray, xf::AbstractVector)
    H = zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = zeros(size(R,1))
    c = 0.5*xf'*Q*xf
    return QuadraticCost(Q, R, H, q, r, c)
end

"""
$(SIGNATURES)
Cost function of the form
``(x-x_f)^T Q (x-x_f)``
Q must be positive semidefinite
"""
function LQRCostTerminal(Qf::AbstractArray,xf::AbstractVector)
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf
    return QuadraticCost(Qf,zeros(0,0),zeros(0,size(Qf,1)),qf,zeros(0),cf)
end

# Cost function methods
function stage_cost(cost::QuadraticCost, x::AbstractVector, u::AbstractVector)
    0.5*x'cost.Q*x + 0.5*u'*cost.R*u + cost.q'x + cost.r'u + cost.c + u'*cost.H*x
end

function stage_cost(cost::QuadraticCost, xN::AbstractVector{T}) where T
    0.5*xN'cost.Q*xN + cost.q'*xN + cost.c
end

function gradient(cost::QuadraticCost, x, u)
    Qx = cost.Q*x + cost.q + cost.H'*u
    Qu = cost.R*u + cost.r + cost.H*x
    return Qx, Qu
end

function hessian(cost::QuadraticCost, x, u)
    Qxx = cost.Q
    Quu = cost.R
    Qux = cost.H
    return Qxx, Quu, Qux
end


# Additional Methods
function Base.show(io::IO, cost::QuadraticCost)
    print(io, "QuadraticCost{...}")
end

import Base: +
"Add two Quadratic costs of the same size"
function +(c1::QuadraticCost, c2::QuadraticCost)
    @assert state_dim(c1) == state_dim(c2)
    @assert control_dim(c1) == control_dim(c2)
    QuadraticCost(c1.Q + c2.Q, c1.R + c2.R, c1.H + c2.H,
                  c1.q + c2.q, c1.r + c2.r, c1.c + c2.c)
end

function copy(cost::QuadraticCost)
    return QuadraticCost(copy(cost.Q), copy(cost.R), copy(cost.H), copy(cost.q), copy(cost.r), copy(cost.c))
end


#
# """
# $(TYPEDEF)
# Cost function of the form
#     ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf
# """
# struct GenericCost <: CostFunction
#     ℓ::Function             # Stage cost
#     ℓf::Function            # Terminal cost
#     expansion::Function     # 2nd order Taylor Series Expansion of the form,  Q,R,H,q,r = expansion(x,u)
#     n::Int                  #                                                     Qf,qf = expansion(xN)
#     m::Int
# end
#
# """
# $(SIGNATURES)
# Create a Generic Cost, specifying the gradient and hessian of the cost function analytically
#
# # Arguments
# * hess: multiple-dispatch function of the form,
#     Q,R,H = hess(x,u) with sizes (n,n), (m,m), (m,n)
#     Qf = hess(xN) with size (n,n)
# * grad: multiple-dispatch function of the form,
#     q,r = grad(x,u) with sizes (n,), (m,)
#     qf = grad(x,u) with size (n,)
#
# """
# function GenericCost(ℓ::Function, ℓf::Function, grad::Function, hess::Function, n::Int, m::Int)
#     @warn "Use GenericCost with caution. It is untested and not likely to work"
#     function expansion(x::Vector{T},u::Vector{T}) where T
#         Q,R,H = hess(x,u)
#         q,r = grad(x,u)
#         return Q,R,H,q,r
#     end
#     expansion(xN) = hess(xN), grad(xN)
#     GenericCost(ℓ,ℓf, expansion, n,m)
# end
#
# """ $(TYPEDEF)
# This is an experimental cost function type that allows a cost function to be evaluated
#     on only a portion of the state and control. Right now, the implementation assumes
#     there is no coupling between the state and control.
#
# It should be noted that for `QuadraticCost`s it is likely more efficient to simply make a new
#     `QuadraticCost` that has zeros in the right places, and then add the cost functions together.
#
# # Constructor
# ```julia
# IndexedCost(cost, ix::UnitRange, iu::UnitRange)
# ```
# """
# struct IndexedCost{iX,iU,C} <: CostFunction
#     cost::C
# end
#
# function IndexedCost(cost::C, ix::UnitRange, iu::UnitRange) where C<:CostFunction
#     if C <: QuadraticCost
#         if norm(cost.H) != 0
#             throw(ErrorException("IndexedCost of functions with x-u coupling not implemented"))
#         end
#     else
#         @warn "IndexedCost will only work for costs without x-u coupling (Qux = 0)"
#     end
#     IndexedCost{ix,iu,C}(cost)
# end
#
# @generated function stage_cost(costfun::IndexedCost{iX,iU}, x::SVector{N}, u::SVector{M}) where {iX,iU,N,M}
#     ix = SVector{length(iX)}(iX)
#     iu = SVector{length(iU)}(iU)
#     quote
#         x0 = x[$ix]
#         u0 = u[$iu]
#         stage_cost(costfun.cost, x0, u0)
#     end
# end
#
# @generated function stage_cost(costfun::IndexedCost{iX,iU}, x::SVector{N}) where {iX,iU,N}
#     ix = SVector{length(iX)}(iX)
#     quote
#         x0 = x[$ix]
#         stage_cost(costfun.cost, x0)
#     end
# end
#
# @generated function gradient(costfun::IndexedCost{iX,iU}, x::SVector{N}, u::SVector{M}) where {iX,iU,N,M}
#     l1x = iX[1] - 1
#     l2x = N-iX[end]
#     l1u = iU[1] - 1
#     l2u = M-iU[end]
#     quote
#         x = x[$iX]
#         u = u[$iU]
#         Qx, Qu = gradient(costfun.cost, x, u)
#         Qx = [@SVector zeros($l1x); Qx; @SVector zeros($l2x)]
#         Qu = [@SVector zeros($l1u); Qu; @SVector zeros($l2u)]
#         return Qx, Qu
#     end
# end
#
#
# @generated function hessian(costfun::IndexedCost{iX,iU}, x::SVector{N}, u::SVector{M}) where {iX,iU,N,M}
#     l1x = iX[1] - 1
#     l2x = N-iX[end]
#     l1u = iU[1] - 1
#     l2u = M-iU[end]
#     quote
#         x = x[$iX]
#         u = u[$iU]
#         Qxx, Quu, Qux  = hessian(costfun.cost, x, u)
#         Qxx1 = Diagonal(@SVector zeros($l1x))
#         Qxx2 = Diagonal(@SVector zeros($l2x))
#         Quu1 = Diagonal(@SVector zeros($l1u))
#         Quu2 = Diagonal(@SVector zeros($l2u))
#
#         Qxx = blockdiag(Qxx1, Qxx, Qxx2)
#         Quu = blockdiag(Quu1, Quu, Quu2)
#         Qux = @SMatrix zeros(M,N)
#         Qxx, Quu, Qux
#     end
# end
#
# function SparseArrays.blockdiag(Qs::Vararg{<:Diagonal})
#     Diagonal(vcat(diag.(Qs)...))
# end
#
# function SparseArrays.blockdiag(Qs::Vararg{<:AbstractMatrix})
#     # WARNING: this is slow and is only included as a fallback
#     cat(Qs...,dims=(1,2))
# end
#
# function change_dimension(cost::CostFunction,n,m)
#     n0,m0 = state_dim(cost), control_dim(cost)
#     ix = 1:n0
#     iu = 1:m0
#     IndexedCost(cost, ix, iu)
# end
#
# function change_dimension(cost::QuadraticCost, n, m)
#     n0,m0 = state_dim(cost), control_dim(cost)
#     @assert n >= n0
#     @assert m >= m0
#
#     ix = 1:n0
#     iu = 1:m0
#
#     Q_ = Diagonal(@SVector zeros(n-n0))
#     R_ = Diagonal(@SVector zeros(m-m0))
#     H1 = @SMatrix zeros(m0, n-n0)
#     H2 = @SMatrix zeros(m-m0, n)
#     q_ = @SVector zeros(n-n0)
#     r_ = @SVector zeros(m-m0)
#     c = cost.c
#
#     # Insert old values
#     Q = blockdiag(cost.Q, Q_)
#     R = blockdiag(cost.R, R_)
#     H = [cost.H H1]
#     H = [H; H2]
#     q = [cost.q; q_]
#     r = [cost.r; r_]
#     QuadraticCost(Q,R,H,q,r,c,checks=false)
# end
