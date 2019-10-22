"""
    TrajectoryOptimization
Primary module for setting up and solving trajectory optimization problems.
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
using SuiteSparse
using StaticArrays
using Logging
using Formatting
using Plots
using BenchmarkTools
using PartedArrays
using Parameters
using Rotations
using BlockArrays
using MathOptInterface
using TimerOutputs

export
    Dynamics,
    Problems

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
    DIRTRELSolver,
    Discrete,
    Continuous,
    Constraint,
    BoundConstraint,
    Equality,
    Inequality,
    ConstraintSet,
    StageConstraintSet,
    TerminalConstraintSet,
    ConstraintSet,
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

# Trajectory Types
Trajectory{T} = Vector{T} where T <: AbstractArray
VectorTrajectory{T} = Vector{Vector{T}} where T <: Real
MatrixTrajectory{T} = Vector{Matrix{T}} where T <: Real
AbstractVectorTrajectory{T} = Vector{V} where {V <: AbstractVector{T}, T <: Real}
DiagonalTrajectory{T} = Vector{Diagonal{T,Vector{T}}} where T <: Real
PartedVecTrajectory{T} = Vector{PartedVector{T,Vector{T}}}
PartedMatTrajectory{T} = Vector{PartedMatrix{T,Matrix{T}}}

include("constraints.jl")
include("constraint_sets.jl")
include("cost.jl")
include("model.jl")
include("integration.jl")
include("utils.jl")
include("objective.jl")
include("problem.jl")
include("solvers.jl")
include("rollout.jl")
include("dynamics.jl")
include("problems.jl")
include("logger.jl")

include("static_problem.jl")
include("static_model.jl")
include("solvers/silqr/silqr_solver.jl")
include("solvers/silqr/silqr_methods.jl")

write_ipopt_options()
end
