abstract type AbstractObjective end

CostTrajectory = Vector{C} where C <: CostFunction

"$(TYPEDEF) Objective: stores stage cost(s) and terminal cost functions"
struct ObjectiveNew <: AbstractObjective
    cost::CostTrajectory
end

function ObjectiveNew(cost::CostFunction,N::Int)
    ObjectiveNew([cost for k = 1:N])
end

"Input requires separate stage and terminal costs (and trajectory length)"
function ObjectiveNew(cost::CostFunction,cost_terminal::CostFunction,N::Int)
    ObjectiveNew([k < N ? cost : cost_terminal for k = 1:N])
end

"Input requires separate stage trajectory and terminal cost"
function ObjectiveNew(cost::CostTrajectory,cost_terminal::CostFunction)
    ObjectiveNew([cost...,cost_terminal])
end

# "Input requires cost function trajectory"
# function ObjectiveNew(cost::CostTrajectory)
#     ObjectiveNew(cost)
# end

import Base.getindex

getindex(obj::ObjectiveNew,i::Int) = obj.cost[i]

"$(TYPEDEF) Augmented Lagrangian Objective: stores stage cost(s) and terminal cost functions"
struct ALObjectiveNew{T} <: AbstractObjective where T
    cost::CostTrajectory
    constraints::ProblemConstraints
    C::PartedVecTrajectory{T}  # Constraint values
    ∇C::PartedMatTrajectory{T} # Constraint jacobians
    λ::PartedVecTrajectory{T}  # Lagrange multipliers
    μ::PartedVecTrajectory{T}  # Penalty Term
    active_set::PartedVecTrajectory{Bool}  # Active set
end

function ALObjectiveNew(cost::CostTrajectory,constraints::ProblemConstraints,N::Int;
        μ_init::T=1.,λ_init::T=0.) where T
    # Get sizes
    n,m = get_sizes(cost)
    C,∇C,λ,μ,active_set = init_constraint_trajectories(constraints,n,m,N)
    ALObjectiveNew{T}(cost,constraint,C,∇C,λ,μ,active_set)
end

function ALObjectiveNew(cost::CostTrajectory,constraints::ProblemConstraints,
        λ::PartedVecTrajectory{T}; μ_init::T=1.) where T
    # Get sizes
    n,m = get_sizes(cost)
    N = length(λ)
    C,∇C,_,μ,active_set = init_constraint_trajectories(constraints,n,m,N)
    ALObjectiveNew{T}(cost,constraint,C,∇C,λ,μ,active_set)
end

getindex(obj::ALObjectiveNew,i::Int) = obj.cost[i]
