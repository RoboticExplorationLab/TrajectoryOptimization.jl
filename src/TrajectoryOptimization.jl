"""
    TrajectoryOptimization
Primary module for setting up and solving trajectory optimization problems with
iterative Linear Quadratic Regulator (iLQR). Module supports unconstrained and
constrained optimization problems. Constrained optimization problems are solved
using Augmented Lagrangian methods. Supports automatic differentiation via the
`ForwardDiff` package by JuliaRobotics.
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
    ConstrainedObjective,
    UnconstrainedObjective,
    ConstrainedResults,
    UnconstrainedResults,
    SolverOptions

# Primary methods
export
    solve,
    rollout!,
    forwardpass!,
    backwardpass!,
    cost,
    max_violation,
    update_objective,
    infeasible_control

include("model.jl")
include("integration.jl")
include("solver.jl")
include("ilqr_algorithm.jl")
include("augmented_lagrange.jl")
include("forensics.jl")
include("dynamics.jl")


end # module
