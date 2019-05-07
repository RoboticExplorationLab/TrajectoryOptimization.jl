"""
    TrajectoryOptimization
Primary module for setting up and solving trajectory optimization problems with
iterative Linear Quadratic Regulator (iLQR). Module supports unconstrained and
constrained optimization problems. Constrained optimization problems are solved
using Augmented Lagrangian methods. Supports automatic differentiation via the
`ForwardDiff` package.
"""
module TrajectoryOptimization

using RigidBodyDynamics
using ForwardDiff
using DocStringExtensions
using Interpolations
using RecipesBase
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using StaticArrays
using Logging
using Formatting
using Plots
using BenchmarkTools
using PartedArrays
using Parameters

export
    Dynamics

# Primary types
export
    Model,
    Solver,
    SolverResults,
    ConstrainedObjective,
    UnconstrainedObjective,
    LQRObjective,
    QuadraticCost,
    LQRCost,
    GenericCost,
    ALCost,
    ConstrainedVectorResults,
    UnconstrainedVectorResults,
    SolverOptions,
    Trajectory

export
    Problem,
    iLQRSolver,
    iLQRSolverOptions,
    AugmentedLagrangianSolver,
    AugmentedLagrangianSolverOptions,
    AugmentedLagrangianProblem,
    Discrete,
    Continuous,
    Constraint,
    TerminalConstraint,
    Equality,
    Inequality,
    ConstraintSet,
    StageConstraintSet,
    TerminalConstraintSet,
    AbstractConstraintSet

export
    rk3,
    rk4,
    midpoint,
    add_constraints!,
    bound_constraint,
    goal_constraint,
    initial_controls!,
    initial_state!


# Primary methods
export
    solve,
    solve!,
    rollout!,
    rollout,
    forwardpass!,
    backwardpass!,
    cost,
    max_violation,
    update_objective,
    infeasible_control,
    line_trajectory,
    evaluate!,
    jacobian!

export
    get_sizes,
    num_constraints,
    get_num_constraints,
    get_num_controls,
    init_results,
    to_array,
    to_dvecs,
    get_N,
    quat2rot,
    sphere_constraint,
    circle_constraint,
    plot_trajectory!,
    plot_vertical_lines!,
    convergence_rate,
    plot_obstacles,
    generate_controller,
    lqr,
    evals,
    reset,
    reset_evals,
    final_time,
    count_constraints,
    inequalities,
    equalities,
    bounds,
    labels,
    terminal,
    stage


# Trajectory Types
Trajectory{T} = Vector{T} where T <: AbstractArray
VectorTrajectory{T} = Vector{Vector{T}} where T <: Real
MatrixTrajectory{T} = Vector{Matrix{T}} where T <: Real
AbstractVectorTrajectory{T} = Vector{V} where {V <: AbstractVector{T}, T <: Real}
DiagonalTrajectory{T} = Vector{Diagonal{T,Vector{T}}} where T <: Real
PartedVecTrajectory{T} = Vector{PartedVector{T,Vector{T}}}
PartedMatTrajectory{T} = Vector{PartedMatrix{T,Matrix{T}}}

include("constraints_type.jl")
include("cost.jl")
include("objective.jl")
include("model.jl")
include("integration.jl")
include("solver.jl")
include("results.jl")
include("results_dircol.jl")
include("backwardpass.jl")
include("forwardpass.jl")
include("constraints.jl")
include("rollout.jl")
#include("newton.jl")
include("infeasible.jl")
include("minimum_time.jl")
include("ilqr_methods.jl")
include("augmented_lagrangian.jl")
include("solve.jl")
include("controller.jl")

include("problem_type.jl")
include("solver_options_new.jl")
include("solvers_new.jl")
include("solvers/direct/direct_solvers.jl")
include("utils.jl")
include("ilqr.jl")
include("altro.jl")
include("backwardpass_new.jl")
include("forward_pass_new.jl")
include("rollout_new.jl")
include("augmented_lagrangian_new.jl")
include("minimum_time_new.jl")
include("dynamics.jl")
include("logger.jl")

using Ipopt

# DIRCOL methods
export
solve_dircol,
gen_usrfun,
DircolResults,
DircolVars,
collocation_constraints,
collocation_constraints!,
cost_gradient,
cost_gradient!,
constraint_jacobian,
constraint_jacobian!,
get_weights,
get_initial_state

export
packZ,
unpackZ

include("dircol.jl")
include("dircol_ipopt.jl")
write_ipopt_options()

# if "Snopt" in keys(Pkg.installed())
#     using Snopt # not safe for precompilation
#     include("dircol_snopt.jl")
# end

end
