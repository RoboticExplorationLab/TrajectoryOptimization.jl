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
    ConstrainedVectorResults,
    UnconstrainedVectorResults,
    SolverOptions,
    Trajectory

# Primary methods
export
    solve,
    rollout!,
    rollout,
    forwardpass!,
    backwardpass!,
    cost,
    max_violation,
    update_objective,
    infeasible_control,
    line_trajectory

export
    get_sizes,
    get_num_constraints,
    get_num_controls,
    init_results,
    to_array,
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
    reset_evals

# Trajectory Types
Trajectory{T} = Vector{T} where T <: AbstractArray
VectorTrajectory{T} = Vector{Vector{T}} where T <: Real
MatrixTrajectory{T} = Vector{Matrix{T}} where T <: Real
DiagonalTrajectory{T} = Vector{Diagonal{T,Vector{T}}} where T <: Real
PartedVecTrajectory{T} = Vector{BlockVector{T,Vector{T}}}
PartedMatTrajectory{T} = Vector{BlockMatrix{T,Matrix{T}}}

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
include("utils.jl")
include("dynamics.jl")
include("logger.jl")
include("controller.jl")

include("problem_type.jl")
include("solver_options_new.jl")
include("solvers_new.jl")
include("ilqr.jl")
include("backwardpass_new.jl")
include("forward_pass_new.jl")
include("rollout_new.jl")
include("augmented_lagrangian_new.jl")

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
