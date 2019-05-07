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
    QuadraticCost,
    LQRCost,
    GenericCost,
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
PartedVecTrajectory{T} = Vector{BlockVector{T,Vector{T}}}
PartedMatTrajectory{T} = Vector{BlockMatrix{T,Matrix{T}}}

include("solver_options.jl")
include("constraints.jl")
include("cost.jl")
include("model.jl")
include("integration.jl")
include("utils.jl")
include("objective.jl")
include("problem.jl")
include("solvers.jl")
include("ilqr.jl")
include("altro.jl")
include("backwardpass.jl")
include("forward_pass.jl")
include("rollout.jl")
include("augmented_lagrangian.jl")
include("minimum_time.jl")
include("infeasible.jl")
include("dynamics.jl")
include("logger.jl")

end
