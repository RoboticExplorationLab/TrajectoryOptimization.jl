import Base.copy
using Parameters

export
    solver_name,
    cost,
    max_violation,
    options,
    stats,
    states,
    controls,
    initial_controls!,
    initial_states!,
    initial_trajectory!,
    rollout!,
    get_trajectory

abstract type AbstractSolver{T} <: MOI.AbstractNLPEvaluator end
abstract type UnconstrainedSolver{T} <: AbstractSolver{T} end
abstract type ConstrainedSolver{T} <: AbstractSolver{T} end
abstract type AbstractSolverOptions{T<:Real} end

abstract type DirectSolver{T} <: ConstrainedSolver{T} end
abstract type DirectSolverOptions{T} <: AbstractSolverOptions{T} end

include("solvers/direct/dircol_ipopt.jl")
include("solvers/direct/dircol_snopt.jl")

@inline options(solver::AbstractSolver) = solver.opts
@inline stats(solver::AbstractSolver) = solver.stats

function cost(solver::AbstractSolver)
    obj = get_objective(solver)
    Z = get_trajectory(solver)
    cost!(obj, Z)
    sum(get_J(obj))
end

function rollout!(solver::AbstractSolver)
    Z = get_trajectory(solver)
    model = get_model(solver)
    x0 = get_initial_state(solver)
    rollout!(model, Z, x0)
end


"Get the state trajectory"
states(solver::AbstractSolver) = [state(z) for z in get_trajectory(solver)]

"Get the control trajectory"
function controls(solver::AbstractSolver)
    N = size(solver)[3]
    Z = get_trajectory(solver)
    [control(Z[k]) for k = 1:N-1]
end

@inline initial_states!(solver::AbstractSolver, X0) = set_states!(get_trajectory(solver), X0)
@inline initial_controls!(solver::AbstractSolver, U0) = set_controls!(get_trajectory(solver), U0)
function initial_trajectory!(solver::AbstractSolver, Z0::Traj)
    Z = get_trajectory(solver)
    for k in eachindex(Z)
        Z[k].z = copy(Z0[k].z)
    end
end

# ConstrainedSolver methods
num_constraints(solver::AbstractSolver) = num_constraints(get_constraints(solver))

function max_violation(solver::ConstrainedSolver, Z::Traj)
    update_constraints!(solver, Z)
    max_violation(solver)
end

function max_violation(solver::ConstrainedSolver)
    conSet = get_constraints(solver)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end

function update_constraints!(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver))
    conSet = get_constraints(solver)
    evaluate!(conSet, Z)
end

function update_active_set!(solver::ConstrainedSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    update_active_set!(conSet, Z, Val(solver.opts.active_set_tolerance))
end

function constraint_jacobian!(solver::ConstrainedSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    jacobian!(conSet, Z)
    return nothing
end
