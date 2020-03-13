import Base.copy
using Parameters

export
    UnconstrainedSolver,
    ConstrainedSolver,
    DirectSolver

export
    solver_name,
    cost,
    max_violation,
    options,
    stats,
    iterations,
    states,
    controls,
    initial_controls!,
    initial_states!,
    initial_trajectory!,
    set_initial_state!,
    get_trajectory,
    get_model,
    get_times,
    initialize!

""" $(TYPEDEF)
Abstract solver for trajectory optimization problems

Any type that inherits from `AbstractSolver` must define the following methods:
```julia
model = get_model(::AbstractSolver)::AbstractModel
obj = get_objective(::AbstractSolver)::AbstractObjective
Z = get_trajectory(::AbstractSolver)::Traj
n,m,N = Base.size(::AbstractSolver)
x0 = get_initial_state(::AbstractSolver)::SVector
solve!(::AbstractSolver)
```
"""
abstract type AbstractSolver{T} <: MOI.AbstractNLPEvaluator end

"$(TYPEDEF) Unconstrained optimization solver. Will ignore
any constraints in the problem"
abstract type UnconstrainedSolver{T} <: AbstractSolver{T} end


"""$(TYPEDEF)
Abstract solver for constrained trajectory optimization problems

In addition to the methods required for `AbstractSolver`, all `ConstrainedSolver`s
    must define the following method
```julia
get_constraints(::ConstrainedSolver)::ConstrainSet
```
"""
abstract type ConstrainedSolver{T} <: AbstractSolver{T} end


""" $(TYPEDEF)
Solve the trajectory optimization problem by computing search directions using the joint
state vector, often solving the KKT system directly.
"""
abstract type DirectSolver{T} <: ConstrainedSolver{T} end

include("solvers/direct/dircol_ipopt.jl")
include("solvers/direct/dircol_snopt.jl")

@inline options(solver::AbstractSolver) = solver.opts
@inline stats(solver::AbstractSolver) = solver.stats
iterations(solver::AbstractSolver) = stats(solver).iterations

function cost(solver::AbstractSolver)
    obj = get_objective(solver)
    Z = get_trajectory(solver)
    cost(obj, Z)
end

function RobotDynamics.rollout!(solver::AbstractSolver)
    Z = get_trajectory(solver)
    model = get_model(solver)
    x0 = get_initial_state(solver)
    rollout!(model, Z, x0)
end

TrajOptCore.set_initial_state!(solver, x0) = copyto!(get_initial_state(solver), x0)

RobotDynamics.states(solver::AbstractSolver) = [state(z) for z in get_trajectory(solver)]

function RobotDynamics.controls(solver::AbstractSolver)
    N = size(solver)[3]
    Z = get_trajectory(solver)
    [control(Z[k]) for k = 1:N-1]
end

@inline TrajOptCore.initial_states!(solver::AbstractSolver, X0) = set_states!(get_trajectory(solver), X0)
@inline TrajOptCore.initial_controls!(solver::AbstractSolver, U0) = set_controls!(get_trajectory(solver), U0)
function TrajOptCore.initial_trajectory!(solver::AbstractSolver, Z0::Traj)
    Z = get_trajectory(solver)
    for k in eachindex(Z)
        Z[k].z = copy(Z0[k].z)
    end
end

@inline get_trajectory(solver::AbstractSolver) = solver.Z
@inline get_times(solver::AbstractSolver) = get_times(get_trajectory(solver))

# ConstrainedSolver methods
TrajOptCore.num_constraints(solver::AbstractSolver) = num_constraints(get_constraints(solver))

function TrajOptCore.max_violation(solver::ConstrainedSolver, Z::Traj)
    update_constraints!(solver, Z)
    max_violation(solver)
end

function TrajOptCore.max_violation(solver::ConstrainedSolver)
    conSet = get_constraints(solver)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end

@inline TrajOptCore.findmax_violation(solver::ConstrainedSolver) =
    findmax_violation(get_constraints(solver))

""" $(SIGNATURES)
Calculate all the constraint values given the trajectory `Z`
"""
function update_constraints!(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver))
    conSet = get_constraints(solver)
    evaluate!(conSet, Z)
end

function TrajOptCore.update_active_set!(solver::ConstrainedSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    update_active_set!(conSet, Z, Val(solver.opts.active_set_tolerance))
end

""" $(SIGNATURES)
Calculate all the constraint Jacobians given the trajectory `Z`
"""
function constraint_jacobian!(solver::ConstrainedSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    jacobian!(conSet, Z)
    return nothing
end


# Logging
function set_verbosity!(opts)
    log_level = opts.log_level
    if opts.verbose
        set_logger()
        Logging.disable_logging(LogLevel(log_level.level-1))
    else
        Logging.disable_logging(log_level)
    end
end

function clear_cache!(opts)
    if opts.verbose
        log_level = opts.log_level
        clear_cache!(global_logger().leveldata[log_level])
    end
end
