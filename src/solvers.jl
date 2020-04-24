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

"""
    AbstractSolver{T} <: MathOptInterface.AbstractNLPEvaluator

Abstract solver for trajectory optimization problems

# Interface
Any type that inherits from `AbstractSolver` must define the following methods:
```julia
model = get_model(::AbstractSolver)::AbstractModel
obj   = get_objective(::AbstractSolver)::AbstractObjective
E     = get_cost_expansion(::AbstractSolver)::QuadraticExpansion  # quadratic error state expansion
Z     = get_trajectory(::AbstractSolver)::Traj
n,m,N = Base.size(::AbstractSolver)
x0    = get_initial_state(::AbstractSolver)::StaticVector
solve!(::AbstractSolver)
```

Optional methods for line search and merit function interface. Note that these do not
    have to return `Traj`
```julia
Z     = get_solution(::AbstractSolver)  # current solution (defaults to get_trajectory)
Z     = get_primals(::AbstractSolver)   # current primals estimate used in the line search
dZ    = get_step(::AbstractSolver)      # current step in the primal variables
```

Optional methods
```julia
opts  = options(::AbstractSolver)       # options struct for the solver. Defaults to `solver.opts`
st    = solver_stats(::AbstractSolver)  # dictionary of solver statistics. Defaults to `solver.stats`
iters = iterations(::AbstractSolver)    #
```
"""
abstract type AbstractSolver{T} <: MOI.AbstractNLPEvaluator end

# Default getters
@inline get_model(solver::AbstractSolver) = solver.model
@inline get_objective(solver::AbstractSolver) = solver.obj
@inline get_cost_expansion(solver::AbstractSolver) = solver.E
@inline get_cost_expansion_error(solver::AbstractSolver) = solver.E
@inline get_error_state_jacobians(solver::AbstractSolver) = solver.G
@inline TrajOptCore.get_trajectory(solver::AbstractSolver) = solver.Z.Z_
@inline get_initial_state(solver::AbstractSolver) = solver.x0

@inline get_solution(solver::AbstractSolver) = get_trajectory(solver)
@inline get_primals(solver::AbstractSolver) = solver.Z̄
@inline get_step(solver::AbstractSolver) = solver.δZ
@inline options(solver::AbstractSolver) = solver.opts
@inline stats(solver::AbstractSolver) = solver.stats
iterations(solver::AbstractSolver) = stats(solver).iterations


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

@inline get_duals(solver::ConstrainedSolver) = get_duals(get_constraints(solver))


""" $(TYPEDEF)
Solve the trajectory optimization problem by computing search directions using the joint
state vector, often solving the KKT system directly.
"""
abstract type DirectSolver{T} <: ConstrainedSolver{T} end

include("solvers/direct/dircol_ipopt.jl")
include("solvers/direct/dircol_snopt.jl")

function TrajOptCore.cost(solver::AbstractSolver, Z=get_trajectory(solver))
    obj = get_objective(solver)
    cost(obj, Z)
end

function TrajOptCore.cost_expansion!(solver::AbstractSolver)
    Z = get_trajectory(solver)
    E, obj = get_cost_expansion(solver), get_objective(solver)
    cost_expansion!(E, obj, Z)
end

TrajOptCore.error_expansion!(solver::AbstractSolver) = error_expansion_uncon!(solver)

""" $(SIGNATURES)
Calculate all the constraint values given the trajectory `Z`
"""
function update_constraints!(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver))
    conSet = get_constraints(solver)
    evaluate!(conSet, Z)
end

function TrajOptCore.update_active_set!(solver::ConstrainedSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    update_active_set!(conSet, Val(solver.opts.active_set_tolerance))
end

""" $(SIGNATURES)
Calculate all the constraint Jacobians given the trajectory `Z`
"""
function constraint_jacobian!(solver::ConstrainedSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    jacobian!(conSet, Z)
    return nothing
end

#--- Error Expansion ---#
function error_expansion_uncon!(solver::AbstractSolver)
    E  = get_cost_expansion_error(solver)
    E0 = get_cost_expansion(solver)
    G  = get_error_state_jacobians(solver)
    error_expansion!(E, E0, get_model(solver), G)
end

function TrajOptCore.error_expansion!(solver::ConstrainedSolver)
    error_expansion_uncon!(solver)
    conSet = get_constraints(solver)
    G = get_error_state_jacobians(solver)
    error_expansion!(conSet, get_model(solver), G)
end

#--- Gradient Norm ---#
function norm_grad(solver::UnconstrainedSolver, recalculate::Bool=true)
    if recalculate
        cost_expansion!(solver)
    end
    norm_grad(get_cost_expansion(solver))
end

function norm_grad(solver::ConstrainedSolver, recalculate::Bool=true)
    conSet = get_constraints(solver)
    if recalculate
        cost_expansion!(solver)
        jacobian
    end
    norm_grad(get_cost_expansion(solver))
end

#--- Rollout ---#
function RobotDynamics.rollout!(solver::AbstractSolver)
    Z = get_trajectory(solver)
    model = get_model(solver)
    x0 = get_initial_state(solver)
    rollout!(model, Z, x0)
end

RobotDynamics.states(solver::AbstractSolver) = [state(z) for z in get_trajectory(solver)]
function RobotDynamics.controls(solver::AbstractSolver)
    N = size(solver)[3]
    Z = get_trajectory(solver)
    [control(Z[k]) for k = 1:N-1]
end

set_initial_state!(solver::AbstractSolver, x0) = copyto!(get_initial_state(solver), x0)

@inline TrajOptCore.initial_states!(solver::AbstractSolver, X0) = set_states!(get_trajectory(solver), X0)
@inline TrajOptCore.initial_controls!(solver::AbstractSolver, U0) = set_controls!(get_trajectory(solver), U0)
function TrajOptCore.initial_trajectory!(solver::AbstractSolver, Z0::Traj)
    Z = get_trajectory(solver)
    for k in eachindex(Z)
        RobotDynamics.set_z!(Z[k], Z0[k].z)
    end
end

# Default getters
@inline RobotDynamics.get_times(solver::AbstractSolver) = RobotDynamics.get_times(get_trajectory(solver))


# Line Search and Merit Functions
"""
    get_primals(solver)
    get_primals(solver, α::Real)

Get the primal variables used during the line search. When called without an `α` it should
return the container where the temporary primal variables are stored. When called with a
step length `α` it returns `z + α⋅dz` where `z = get_solution(solver)` and `dz = get_step(solver)`.
"""
function get_primals(solver::AbstractSolver, α)
    z = get_solution(solver)
    z̄ = get_primals(solver)
    dz = get_step(solver)
    z̄ .= z .+ α*dz
end

# Constrained solver
TrajOptCore.num_constraints(solver::AbstractSolver) = num_constraints(get_constraints(solver))

function TrajOptCore.max_violation(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver); recalculate=true)
    conSet = get_constraints(solver)
    if recalculate
        evaluate!(conSet, Z)
    end
    max_violation(conSet)
end

function TrajOptCore.norm_violation(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver); recalculate=true, p=2)
    conSet = get_constraints(solver)
    if recalculate
        evaluate!(conSet, Z)
    end
    TrajOptCore.norm_violation(conSet, p)
end

@inline findmax_violation(solver::ConstrainedSolver) =
    findmax_violation(get_constraints(solver))

function second_order_correction!(solver::ConstrainedSolver)
    conSet = get_constraints(solver)
	Z = get_primals(solver)     # get the current value of z + α⋅δz
	evaluate!(conSet, Z)        # update constraints at new step
	throw(ErrorException("second order correction not implemented yet..."))
end

"""
	cost_dgrad(solver, Z, dZ; recalculate)

Return the scalar directional gradient of the cost evaluated at `Z` in the direction of `dZ`,
where `Z` and `dZ` are the types returned by `get_primals(solver)` and `get_step(solver)`,
and must be able to be converted to a vector of `KnotPoint`s via `Traj()`.
"""
function cost_dgrad(solver::AbstractSolver, Z=get_primals(solver), dZ=get_step(solver);
		recalculate=true)
	E = get_cost_expansion(solver)
	if recalculate
		obj = get_objective(solver)
		cost_gradient!(E, obj, Traj(Z))
	end
	TrajOptCore.dgrad(E, Traj(dZ))
end

"""
	norm_dgrad(solver, Z, dZ; recalculate, p)

Calculate the directional derivative of `norm(c(Z), p)` in the direction of `dZ`, where
`c(Z)` is the vector of constraints evaluated at `Z`.
`Z` and `dZ` are the types returned by `get_primals(solver)` and `get_step(solver)`,
and must be able to be converted to a vector of `KnotPoint`s via `Traj()`.
"""
function norm_dgrad(solver::AbstractSolver, Z=get_primals(solver), dZ=get_step(solver);
		recalculate=true, p=1)
    conSet = get_constraints(solver)
	if recalculate
		Z_ = Traj(Z)
		evaluate!(conSet, Z_)
		jacobian!(conSet, Z_)
	end
    Dc = TrajOptCore.norm_dgrad(conSet, Traj(dZ), 1)
end


"""
	dhess(solver, Z, dZ; recalculate)

Calculate the scalar 0.5*dZ'G*dZ where G is the hessian of cost, evaluating the cost hessian
at `Z`.
"""
function cost_dhess(solver::AbstractSolver, Z=TrajOptCore.get_primals(solver),
		dZ=TrajOptCore.get_step(solver); recalculate=true)
	E = get_cost_expansion(solver)
	if recalculate
		obj = get_objective(solver)
		cost_hessian!(E, obj, Traj(Z))
	end
	TrajOptCore.dhess(solver.E, Traj(dZ))
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
