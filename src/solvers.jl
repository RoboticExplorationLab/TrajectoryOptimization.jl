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

# include("solvers/ilqr/ilqr_solver.jl")
# include("solvers/ilqr/ilqr_methods.jl")
# include("solvers/ilqr/backward_pass.jl")
# include("solvers/ilqr/forward_pass.jl")

# include("solvers/augmented_lagrangian/augmented_lagrangian_solver.jl")
# include("solvers/augmented_lagrangian/augmented_lagrangian_methods.jl")


# include("solvers/direct/direct_solvers.jl")
# include("solvers/direct/sequential_newton.jl")
# include("solvers/direct/dircol.jl")
include("solvers/direct/dircol_ipopt.jl")
include("solvers/direct/dircol_snopt.jl")
# include("solvers/direct/moi.jl")
# include("solvers/direct/sequential_newton_solve.jl")
# include("solvers/direct/projected_newton.jl")

# include("solvers/altro/altro_solver.jl")
# include("solvers/altro/altro_methods.jl")
# include("solvers/altro/infeasible.jl")
# include("solvers/altro/minimum_time.jl")

# include("solvers/direct/primals_mintime.jl")
# include("solvers/direct/direct_solvers_mintime.jl")
# include("solvers/direct/dircol_mintime.jl")
# include("solvers/direct/moi_mintime.jl")

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




# Get name of solver as a string
# solver_name(::iLQRSolverOptions) = "iLQR"
# solver_name(::ALTROSolverOptions) = "ALTRO"
# solver_name(opts::DIRCOLSolverOptions) = string(optimizer_name(opts.nlp))
# solver_name(opts::AugmentedLagrangianSolverOptions) = "AL-" * solver_name(opts.opts_uncon)
# solver_name(solver::AbstractSolver) = solver_name(solver.opts)

#
# # Solver interface
# """$(SIGNATURES)
# Solve trajectory optimization problem `prob` using `solver`.
#     The problem will be modified in place, with the solution stored in `prob.X` and `prob.U`.
#     The solver will also be modified, and may either return the same solver or a new one that may or may not be the same type as the one given.
# """
# solve!(prob::Problem{T,D}, solver::AbstractSolver{T}) where {T<:AbstractFloat,D<:DynamicsType} =
#     error("Cannot solve with `AbstractSolver`")
#
# "```
# AbstractSolver(prob::Problem, opts::AbstractSolverOptions)::AbsractSolver
# ```
# Create a solver, with the type specified by the type of the solver options `opts` "
# AbstractSolver(::Problem{T,D}, ::AbstractSolverOptions{T}) where {T<:AbstractFloat,D<:DynamicsType} =
#     error("Can't create an Abstract Solver without knowing the type of the Solver Options")
#
# "```
# reset!(solver::AbstractSolver)
# ```
# Reset the solver, including initial values for the fields and solve statistics"
# reset!(::AbstractSolver) = nothing
#
# "```
# copy(solver::AbstractSolver)::AbstractSolver
# ```
# Create a copy of the solver with zero associated memory between the two solvers"
# copy(::AbstractSolver)::AbstractSolver = error("Cannot copy `AbstractSolver`")
#
# "```
# size(solver::AbstractSolver)::NTuple{3,Int}
# ```
#  Return the number of controls (n), number of states (m), and the number of knot points (N) as a tuple, i.e. (n,m,N)"
# size(::AbstractSolver)::NTuple{3,Int} = error("`AbstractSolver` has no size")
#
#
#
#
# # Generic methods for calling solve
# """```
# solve!(prob, opts)::AbstractSolver
# ```
# Solve the trajectory optimization problem `prob` using the solver specified by solver options `opts`.
#     The problem will be modified in place, storing the solution in `prob.X` and `prob.U`.
# """
# function solve!(prob::Problem{T,D}, opts::AbstractSolverOptions{T}) where {T<:AbstractFloat, D<:DynamicsType}
#     solver = AbstractSolver(prob, opts)
#     solve!(prob, solver)
# end
#
# """```
# solve(prob, opts)::Tuple{Problem,AbstractSolver}
# ```
# Solve the trajectory optimization problem `prob` using the solver specified by solver options `opts`.
#     The problem will not be modified in place,
#     and will instead return a new problem with the solution in `prob.X` and `prob.U`,
#     along with the solver.
# """
# function solve(prob::Problem{T,D}, opts::AbstractSolverOptions{T}) where {T<:AbstractFloat, D<:DynamicsType}
#     prob0 = copy(prob)
#     solver = solve!(prob0, opts)
#     return prob0, solver
# end
#
# """ ```
# solve(prob, solver)::Tuple{Problem,AbstractSolver}
# ```
# Solve the trajectory optimization problem `prob` using `solver`.
#     The problem will not be modified in place,
#     and will instead return a new problem with the solution in `prob.X` and `prob.U`,
#     along with the solver. The solver will be modified in place,
#         and may or may not be the same solver returned.
# """
# function solve(prob::Problem{T,D}, solver::AbstractSolver{T}) where {T<:AbstractFloat, D<:DynamicsType}
#     prob0 = copy(prob)
#     solver = solve!(prob0, solver)
#     return prob0, solver
# end
#
# jacobian!(prob::Problem{T,Continuous}, solver::AbstractSolver) where T = jacobian!(solver.∇F, prob.model, prob.X, prob.U)
# jacobian!(prob::Problem{T,Discrete},   solver::AbstractSolver) where T = jacobian!(solver.∇F, prob.model, prob.X, prob.U, get_dt_traj(prob))
#
#
# function check_convergence_criteria(opts_uncon::AbstractSolverOptions{T},cost_tolerance::T,gradient_norm_tolerance::T) where T
#     if opts_uncon.cost_tolerance != cost_tolerance
#         @warn "Augmented Lagrangian cost tolerance overriding unconstrained solver option\n >>cost tolerance=$cost_tolerance"
#     end
#
#     if opts_uncon.gradient_norm_tolerance != gradient_norm_tolerance
#         @warn "Augmented Lagrangian gradient norm tolerance overriding unconstrained solver option\n >>gradient norm tolerance=$gradient_norm_tolerance"
#     end
#     return nothing
# end
