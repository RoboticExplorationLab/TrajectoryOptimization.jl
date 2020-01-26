############################################################################################
#                              COST EXPANSIONS                                             #
############################################################################################
"""
$(TYPEDEF)
Store the terms of the 2nd order expansion for the entire trajectory
"""
struct CostExpansion{T,N,M,A1,A2,A3}
    x::Vector{SVector{N,T}}
    u::Vector{SVector{M,T}}
    xx::Vector{A1}
    uu::Vector{A2}
    ux::Vector{A3}
end

function CostExpansion(n,m,N)
    Cxx = n^2 > MAX_ELEM ? [zeros(n,n) for k = 1:N] : [@SMatrix zeros(n,n) for k = 1:N]
    Cuu = m^2 > MAX_ELEM ? [zeros(m,m) for k = 1:N] : [@SMatrix zeros(m,m) for k = 1:N]
    Cux = n*m > MAX_ELEM ? [zeros(m,n) for k = 1:N] : [@SMatrix zeros(m,n) for k = 1:N]
    CostExpansion(
        [@SVector zeros(n) for k = 1:N],
        [@SVector zeros(m) for k = 1:N],
        Cxx,
        Cuu,
        Cux )
end

function Base.getindex(Q::CostExpansion, k::Int)
    return (x=Q.x[k], u=Q.u[k], xx=Q.xx[k], uu=Q.uu[k], ux=Q.ux[k])
end


############################################################################################
#                              OBJECTIVES                                                  #
############################################################################################

abstract type AbstractObjective end
Base.length(obj::AbstractObjective) = length(obj.cost)

"""```
cost(obj::Objective, Z::Traj)::Float64
cost(obj::Objective, dyn_con::DynamicsConstraint{Q}, Z::Traj)
```
Evaluate the cost for a trajectory.
Calculate the cost gradient for an entire trajectory. If a dynamics constraint is given,
    use the appropriate integration rule, if defined.
"""
function cost(obj::AbstractObjective, Z)
    cost!(obj, Z)
    J = get_J(obj)
    return sum(J)
end


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
function LQRObjective(Q::AbstractArray, R::AbstractArray, Qf::AbstractArray,
        xf::AbstractVector, N::Int; checks=true, uf=zeros(size(R,1)))
    H = zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = QuadraticCost(Q, R, H, q, r, c, checks=checks)
    ℓN = QuadraticCost(Qf, qf, cf, check=checks)

    Objective(ℓ, ℓN, N)
end

function LQRObjective(
        Q::Union{Diagonal{T,<:SVector{n}},SMatrix{n,n}},
        R::Union{Diagonal{T,<:SVector{m}},SMatrix{m,m}},
        Qf::AbstractArray, xf::AbstractVector, N::Int;
        uf=(@SVector zeros(m)),
        checks=true) where {T,n,m}
    H = @SMatrix zeros(m,n)
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = QuadraticCost(Q, R, H, q, r, c, checks=checks)

    ℓN = QuadraticCost(Qf, R, H, qf, r, cf, checks=checks)

    Objective(ℓ, ℓN, N)
end
