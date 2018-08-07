# """
#     iLQR
# Primary module for setting up and solving trajectory optimization problems with
# iterative Linear Quadratic Regulator (iLQR). Module supports unconstrained and
# constrained optimization problems. Constrained optimization problems are solved
# using Augmented Lagrangian methods. Supports automatic differentiation for computing dynamics Jacobians via the
# `ForwardDiff` package.
# """
# module iLQR
#     using RigidBodyDynamics
#     using ForwardDiff
#     using DocStringExtensions
#
#     # Primary types
#     export
#         Model,
#         Solver,
#         Objective,
#         SolverOptions
#
#     # Primary methods
#     export
#         solve,
#         rollout!,
#         forwardpass!,
#         backwardpass!,
#         cost,
#         max_violation,
#         bias
#
#     include("model.jl")
#     include("integration.jl")
#     include("solver.jl")
#     include("ilqr_algorithm.jl")
#     include("augmented_lagrange.jl")
#     include("forensics.jl")
# end
