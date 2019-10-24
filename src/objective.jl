abstract type AbstractObjective end

Base.length(obj::AbstractObjective) = length(obj.cost)


"""$(TYPEDEF)
Objective: stores stage cost(s) and terminal cost functions
Constructors:
```julia
Objective(cost, N)
Objective(cost, cost_term, N)
Objective(costs::Vector{<:CostFunction}, cost_term)
```
"""
struct Objective{C} <: AbstractObjective
    cost::Vector{C}
    J::Vector{Float64}
end

function Objective(cost::CostFunction,N::Int)
    Objective([cost for k = 1:N], zeros(N))
end

function Objective(cost::CostFunction,cost_terminal::CostFunction,N::Int)
    Objective([k < N ? cost : cost_terminal for k = 1:N], zeros(N))
end

function Objective(cost::CostTrajectory,cost_terminal::CostFunction)
    Objective([cost...,cost_terminal], zeros(N))
end

import Base.getindex

getindex(obj::Objective,i::Int) = obj.cost[i]

"Allow iteration"
Base.iterate(obj::Objective, start=1) = Base.iterate(obj.cost, start)

Base.show(io::IO, obj::Objective{C}) where C = print(io,"Objective")

"""```julia
cost(obj::Objective, X::Vector, U::Vector, dt::Vector)
```
Calculate cost over entire state and control trajectories
"""
function cost(obj::Objective, X::AbstractVectorTrajectory, U::AbstractVectorTrajectory, dt::Vector)
    N = length(X)
    J = 0.0
    for k = 1:N-1
        J += stage_cost(obj[k],X[k],U[k],dt[k])
    end
    J += stage_cost(obj[N],X[N])
    return J
end

"$(SIGNATURES) Compute the second order Taylor expansion of the cost for the entire trajectory"
function cost_expansion!(Q::ExpansionTrajectory{T}, obj::Objective,
        X::AbstractVectorTrajectory{T}, U::AbstractVectorTrajectory{T}, dt::Vector{T}) where T
    cost_expansion!(Q,obj.cost,X,U,dt)
end

function cost_expansion!(Q::ExpansionTrajectory{T}, c::CostTrajectory,
        X::AbstractVectorTrajectory{T}, U::AbstractVectorTrajectory{T}, dt::Vector{T}) where T
    N = length(X)
    for k = 1:N-1
        cost_expansion!(Q[k],c[k],X[k],U[k],dt[k])
    end
    cost_expansion!(Q[N],c[N],X[N])
end

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
