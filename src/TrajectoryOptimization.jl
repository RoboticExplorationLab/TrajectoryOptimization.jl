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

include("model.jl")
include("integration.jl")
#include("solver_options.jl")
include("solver.jl")
include("results.jl")
include("solve_sqrt.jl")
include("ilqr_algorithm.jl")
include("augmented_lagrange.jl")
include("utils.jl")
include("dynamics.jl")
end
