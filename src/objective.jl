############################################################################################
#                              COST EXPANSIONS                                             #
############################################################################################
"""
$(TYPEDEF)
Store the terms of the 2nd order expansion for the entire trajectory
"""
struct CostExpansion{T,N,M,L1,L2,L3}
    x::Vector{SVector{N,T}}
    u::Vector{SVector{M,T}}
    xx::Vector{SMatrix{N,N,T,L1}}
    uu::Vector{SMatrix{M,M,T,L2}}
    ux::Vector{SMatrix{M,N,T,L3}}
end

function CostExpansion(n,m,N)
    CostExpansion(
        [@SVector zeros(n) for k = 1:N],
        [@SVector zeros(m) for k = 1:N],
        [@SMatrix zeros(n,n) for k = 1:N],
        [@SMatrix zeros(m,m) for k = 1:N],
        [@SMatrix zeros(m,n) for k = 1:N] )
end

function Base.getindex(Q::CostExpansion, k::Int)
    return (x=Q.x[k], u=Q.u[k], xx=Q.xx[k], uu=Q.uu[k], ux=Q.ux[k])
end


############################################################################################
#                              OBJECTIVES                                                  #
############################################################################################

abstract type AbstractObjective end
Base.length(obj::AbstractObjective) = length(obj.cost)


"""$(TYPEDEF)
Objective: stores stage cost(s) and terminal cost functions

Constructors:
```julia
Objective(cost, N)
Objective(cost, cost_term, N)
Objective(costs::Vector{<:CostFunction}, cost_term)
Objective(costs::Vector{<:CostFunction})
```
"""
struct Objective{C} <: AbstractObjective
    cost::Vector{C}
    J::Vector{Float64}
end

# Constructors
function Objective(cost::CostFunction,N::Int)
    Objective([cost for k = 1:N], zeros(N))
end

function Objective(cost::CostFunction,cost_terminal::CostFunction,N::Int)
    Objective([k < N ? cost : cost_terminal for k = 1:N], zeros(N))
end

function Objective(cost::Vector{<:CostFunction},cost_terminal::CostFunction)
    N = length(cost) + 1
    Objective([cost...,cost_terminal], zeros(N))
end

function Objective(cost::Vector{<:CostFunction})
    N = length(cost)
    Objective(cost, zeros(N))
end

# Methods
"Get the vector of costs at each knot point. `sum(get_J(obj))` is equal to the cost"
get_J(obj::Objective) = obj.J

Base.copy(obj::Objective) = Objective(copy(obj.cost), copy(obj.J))

Base.getindex(obj::Objective,i::Int) = obj.cost[i]

Base.iterate(obj::Objective, start=1) = Base.iterate(obj.cost, start)

Base.show(io::IO, obj::Objective{C}) where C = print(io,"Objective")

# Convenience constructors
@doc raw"""```julia
LQRObjective(Q, R, Qf, xf, N)
```
Create an objective of the form
`` (x_N - x_f)^T Q_f (x_N - x_f) + \sum_{k=0}^{N-1} (x_k-x_f)^T Q (x_k-x_f) + u_k^T R u_k``
"""
function LQRObjective(Q::AbstractArray, R::AbstractArray, Qf::AbstractArray, xf::AbstractVector,N::Int)
    H = zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = zeros(size(R,1))
    c = 0.5*xf'*Q*xf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = QuadraticCost(Q, R, H, q, r, c)
    ℓN = QuadraticCost(Qf, qf, cf)

    Objective(ℓ, ℓN, N)
end

function LQRObjective(Q::Union{Diagonal{T,S},SMatrix}, R::AbstractArray, Qf::AbstractArray, xf::AbstractVector,N::Int) where {T,S<:SVector}
    n,m = size(Q,1), size(R,1)
    H = @SMatrix zeros(m,n)
    q = -Q*xf
    r = @SVector zeros(m)
    c = 0.5*xf'*Q*xf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = QuadraticCost(Q, R, H, q, r, c)

    ℓN = QuadraticCost(Qf, R, H, qf, r, cf)

    Objective(ℓ, ℓN, N)
end
