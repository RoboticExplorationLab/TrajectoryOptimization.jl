
############################################################################################
#                              OBJECTIVES                                                  #
############################################################################################

abstract type AbstractObjective end
Base.length(obj::AbstractObjective) = length(obj.cost)
state_dim(obj::AbstractObjective, k::Integer) = throw(ErrorException("state_dim not implemented"))
control_dim(obj::AbstractObjective, k::Integer) = throw(ErrorException("control_dim not implemented"))
get_J(obj::AbstractObjective) = throw(ErrorException("get_J not implemented"))



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
    const_grad::BitVector
    const_hess::BitVector
    diffmethod::Vector{RD.DiffMethod}
    function Objective(cost::Vector{C}; 
                       diffmethod=RD.default_diffmethod.(cost)
    ) where C <: CostFunction
        N = length(cost)
        J = zeros(N)
        grad = zeros(Bool,N)
        hess = zeros(Bool,N)
        if diffmethod isa DiffMethod 
            diffmethod = fill(diffmethod, N)
        end
        new{C}(cost, J, grad, hess, diffmethod)
    end
end

state_dim(obj::Objective, k::Integer) = state_dim(obj.cost[k])
control_dim(obj::Objective, k::Integer) = control_dim(obj.cost[k])
RD.dims(obj::Objective) = state_dim.(obj.cost), control_dim.(obj.cost)
# Base.size(obj::Objective) = (state_dim(obj), control_dim(obj))
@inline ExpansionCache(obj::Objective) = ExpansionCache(obj[1])

import Base.size
@deprecate size(obj::Objective) RobotDynamics.dims(obj)
@deprecate state_dim(obj::Objective) state_dim(obj[1]) 
@deprecate control_dim(obj::Objective) control_dim(obj, 1)

"""
    is_quadratic(obj::Objective)

Only valid for a cost expansion, i.e. an objective containing the 2nd order expansion of 
    another objective. Determines if the original objective is a quadratic function, or 
    in other words, if the hessian of the objective is constant. 

For example, if the original cost function is an augmented Lagrangian cost function, the
    result will return true only if all constraints are linear.
"""
is_quadratic(obj::Objective) = all(obj.const_hess)

# Constructors
function Objective(cost::CostFunction, N::Int; kwargs...)
    Objective([cost for k = 1:N]; kwargs...)
end

function Objective(cost::CostFunction, cost_terminal::CostFunction, N::Int; kwargs...)
    stage, term = promote(cost, cost_terminal)
    Objective([k < N ? stage : term for k = 1:N]; kwargs...)
end

function Objective(cost::Vector{<:CostFunction}, cost_terminal::CostFunction; kwargs...) 
    N = length(cost) + 1
    Objective([cost...,cost_terminal]; kwargs...)
end

"""
	cost(obj::Objective, Z::SampledTrajectory)
	cost(obj::Objective, dyn_con::DynamicsConstraint{Q}, Z::SampledTrajectory)

Evaluate the cost for a trajectory. If a dynamics constraint is given,
    use the appropriate integration rule, if defined.
"""
function cost(obj::Objective, Z::SampledTrajectory{<:Any,<:Any,<:AbstractFloat})
    cost!(obj, Z)
    J = get_J(obj)
    return sum(J)
end

# ForwardDiff-able method
function cost(obj::Objective, Z::SampledTrajectory{<:Any,<:Any,T}) where T
    J = zero(T)
    for k = 1:length(obj)
        J += RD.evaluate(obj[k], Z[k])
    end
    return J
end

"Evaluate the cost for a trajectory (non-allocating)"
@inline function cost!(obj::Objective, Z::SampledTrajectory)
    map!(RD.evaluate, obj.J, obj.cost, Z.data)
end

# Methods
"Get the vector of costs at each knot point. `sum(get_J(obj))` is equal to the cost"
get_J(obj::Objective) = obj.J

Base.copy(obj::Objective) = Objective(copy.(obj.cost); diffmethod=copy(obj.diffmethod))

Base.getindex(obj::Objective,i::Int) = obj.cost[i]

@inline Base.firstindex(obj::Objective) = firstindex(obj.cost)
@inline Base.lastindex(obj::Objective) = lastindex(obj.cost)
@inline Base.iterate(obj::Objective, start=1) = Base.iterate(obj.cost, start)
@inline Base.eltype(obj::Objective) = eltype(obj.cost)
@inline Base.length(obj::Objective) = length(obj.cost)
Base.IteratorSize(obj::Objective) = Base.HasLength()
Base.eachindex(obj::Objective) = Base.OneTo(length(obj))

Base.show(io::IO, obj::Objective{C}) where C = print(io,"Objective")


# Convenience constructors
@doc raw"""
    LQRObjective(Q, R, Qf, xf, N)

Create an objective of the form
`` (x_N - x_f)^T Q_f (x_N - x_f) + \sum_{k=0}^{N-1} (x_k-x_f)^T Q (x_k-x_f) + u_k^T R u_k``

Where `eltype(obj) <: DiagonalCost` if `Q`, `R`, and `Qf` are
    `Union{Diagonal{<:Any,<:StaticVector}}, <:StaticVector}`
"""
function LQRObjective(Q::AbstractArray, R::AbstractArray, Qf::AbstractArray,
        xf::AbstractVector, N::Int; checks=true, diffmethod=UserDefined(), 
        uf=@SVector zeros(size(R,1))
)
    @assert size(Q,1) == length(xf)
    @assert size(Qf,1) == length(xf)
    @assert size(R,1) == length(uf)
    n = size(Q,1)
    m = size(R,1)
    H = SizedMatrix{m,n}(zeros(m,n))
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = QuadraticCost(Q, R, H, q, r, c, checks=checks)
    ℓN = QuadraticCost(Qf, R, H, qf, r, cf, checks=false, terminal=true)

    Objective(ℓ, ℓN, N)
end

function LQRObjective(
        Q::Union{<:Diagonal, <:AbstractVector},
        R::Union{<:Diagonal, <:AbstractVector},
        Qf::Union{<:Diagonal, <:AbstractVector},
        xf::AbstractVector, N::Int;
        diffmethod::DiffMethod = UserDefined(),
        uf=(@SVector zeros(size(R,1))),
        checks=true)
    n,m = size(Q,1), size(R,1)
    @assert size(Q,1) == length(xf)
    @assert size(Qf,1) == length(xf)
    @assert size(R,1) == length(uf)
    Q,R,Qf = Diagonal(Q), Diagonal(R), Diagonal(Qf)
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = DiagonalCost(Q, R, q, r, c, checks=checks, terminal=false)

    ℓN = DiagonalCost(Qf, R, qf, r, cf, checks=false, terminal=true)

    Objective(ℓ, ℓN, N)
end

"""
    TrackingObjective(Q, R, Z; [Qf])

Generate a quadratic objective that tracks the reference trajectory specified by `Z`.
"""
function TrackingObjective(Q,R,Z::SampledTrajectory; Qf=Q)
    costs = map(Z) do z
        LQRCost(Q, R, state(z), control(z))
    end
    costs[end] = LQRCost(Qf, R, state(Z[end]))
    Objective(costs)
end

"""
    update_trajectory!(obj, Z, [start=1])

For use with a tracking-style trajectory (see [`TrackingObjective`](@ref)).
Update the costs to track the new trajectory `Z`. The `start` parameter specifies the 
index of reference trajectory that should be used as the starting point of the reference 
tracked by the objective. This is useful when a single, long time-horizon trajectory is given
but the optimization only tracks a portion of the reference at each solve (e.g. MPC).
"""
function update_trajectory!(obj::Objective{<:QuadraticCostFunction}, Z::SampledTrajectory, start=1)
    inds = (start-1) .+ (1:length(obj))
    for (i,k) in enumerate(inds)
        set_LQR_goal!(obj[i], state(Z[k]), control(Z[k]))
    end
end