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


const level_priorities = Dict(:verbose=>1,:debug=>2,:info=>3,:critical=>4,:none=>Inf)
const debug_level = :critical  # (:verbose, :debug, :info, :critical, :none)

export
    Dynamics


# Primary types
export
    Model,
    Solver,
    SolverResults,
    ConstrainedObjective,
    UnconstrainedObjective,
    ConstrainedResults,
    UnconstrainedResults,
    SolverOptions

# Primary methods
export
    solve,
    solve_al,
    rollout!,
    forwardpass!,
    backwardpass!,
    cost,
    max_violation,
    update_objective,
    infeasible_control,
    line_trajectory

export
    get_sizes,
    get_N,
    eul2quat,
    quat2rot,
    eul2rot,
    sphere_constraints,
    circle_constraints

include("model.jl")
include("integration.jl")
include("solver.jl")
include("results.jl")
include("ilqr_algorithm.jl")
include("ilqr_methods.jl")
include("solve.jl")
include("utils.jl")
include("dynamics.jl")

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
#
# if check_snopt_installation()
#     # using Snopt # not safe for precompilation
#     include("dircol_snopt.jl")
# end

function set_debug_level(level::Symbol)
    global debug_level
    if level ∈ keys(level_priorities)
        debug_level = level
    else
        warn("Debug level not recognized")
    end
    return nothing
end
end
