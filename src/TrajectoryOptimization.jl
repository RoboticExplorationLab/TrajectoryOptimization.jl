"""
    TrajectoryOptimization
Primary module for setting up and solving trajectory optimization problems.
"""
module TrajectoryOptimization

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
using BenchmarkTools
using Parameters
using Rotations
using MathOptInterface
using Quaternions
using UnsafeArrays

const MOI = MathOptInterface
const MAX_ELEM = 170

import Base.copy

export
    Dynamics,
    Problems,
    Controllers

export
    state_dim,
    control_dim

# Primary types
export
    Model,
    QuadraticCost,
    LQRCost,
    LQRObjective,
    GenericCost,
    Trajectory

export
    Problem,
    iLQRSolver,
    iLQRSolverOptions,
    AugmentedLagrangianSolver,
    AugmentedLagrangianSolverOptions,
    AugmentedLagrangianProblem,
    ALTROSolverOptions,
    DIRCOLSolver,
    DIRCOLSolverOptions,
    ProjectedNewtonSolver,
    ProjectedNewtonSolverOptions,
    SequentialNewtonSolver,
    Discrete,
    Continuous,
    Constraint,
    BoundConstraint,
    Equality,
    Inequality,
    Objective,
    Constraints

export
    rk3,
    rk4,
    midpoint,
    add_constraints!,
    goal_constraint,
    initial_controls!,
    initial_state!,
    circle_constraint,
    sphere_constraint


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
    max_violation_direct,
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
    get_N,
    to_dvecs,
    quat2rot,
    sphere_constraint,
    circle_constraint,
    plot_trajectory!,
    plot_vertical_lines!,
    convergence_rate,
    plot_obstacles,
    evals,
    reset,
    reset_evals,
    final_time,
    total_time,
    count_constraints,
    inequalities,
    equalities,
    bounds,
    labels,
    terminal,
    stage,
    interp_rows

# Static methods
export
    convertProblem

include("utils.jl")
include("rotations.jl")
include("logger.jl")
include("knotpoint.jl")
include("expansions.jl")
include("model.jl")
include("costfunctions.jl")
include("objective.jl")
include("solver_opts.jl")
include("solvers.jl")
include("abstract_constraint.jl")
include("constraints.jl")
include("dynamics_constraints.jl")
include("integration.jl")
include("dynamics.jl")

include("cost.jl")
include("static_methods.jl")
include("constraint_vals.jl")
include("constraint_sets.jl")
include("problem.jl")
# include("solvers/silqr/silqr_solver.jl")
# include("solvers/silqr/silqr_methods.jl")
include("solvers/ilqr/ilqr.jl")
include("solvers/ilqr/ilqr_solve.jl")
include("solvers/ilqr/backwardpass.jl")
include("solvers/ilqr/rollout.jl")
include("solvers/augmented_lagrangian/sal_solver.jl")
include("solvers/augmented_lagrangian/sal_methods.jl")
include("solvers/direct/primals.jl")
include("solvers/direct/pn.jl")
include("solvers/direct/pn_methods.jl")
include("solvers/altro/altro_solver.jl")

include("solvers/direct/moi.jl")
include("solvers/direct/copy_blocks.jl")
include("solvers/direct/direct_constraints.jl")

include("problems.jl")
include("controllers.jl")

write_ipopt_options()
Logging.global_logger(default_logger(true))
end
