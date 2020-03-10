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
using RobotDynamics
using DifferentialRotations
using TrajOptCore

import Dynamics: Implicit, Explicit, AbstractKnotPoint, DEFAULT_Q, StaticKnotPoint
import TrajOptCore: DynamicsVals, num_constraints, get_J, cost_expansion!, error_expansion,
    max_violation!, max_penalty!, initial_trajectory!, change_dimension
import TrajOptCore: cost, cost!, get_constraints, get_objective, get_model  # extended
import Dynamics: state_diff

# modules
export
    Problems

#~~~~ Re-export TrajOptCore ~~~~#
export
    Problem,
    Objective,
    LQRObjective,
    LQRCost,
    QuadraticCost,
    initial_states!,
    initial_controls!

# constraints
export
    AbstractConstraint,
    ConstraintSet,
    GoalConstraint,
    BoundConstraint,
    NormConstraint,
    CircleConstraint,
    SphereConstraint,
    LinearConstraint,
    add_constraint!,
    max_violation

# dynamics and integration
export
    RK2, RK3, RK4, HermiteSimpson

# solvers
export
    solve!

const MOI = MathOptInterface


include("utils.jl")
# include("rotations.jl")
include("logger.jl")
# include("expansions.jl")
include("infeasible_model.jl")
# include("costfunctions.jl")
# include("objective.jl")
include("solver_opts.jl")
include("solvers.jl")
# include("abstract_constraint.jl")
# include("constraints.jl")
# include("dynamics_constraints.jl")
# include("integration.jl")
# include("dynamics.jl")

# include("cost.jl")
# include("static_methods.jl")
# include("constraint_vals.jl")
# include("constraint_sets.jl")
# include("problem.jl")
# include("solvers/silqr/silqr_solver.jl")
# include("solvers/silqr/silqr_methods.jl")
include("solvers/ilqr/ilqr.jl")
include("solvers/ilqr/ilqr_solve.jl")
include("solvers/ilqr/backwardpass.jl")
include("solvers/ilqr/rollout.jl")
include("solvers/augmented_lagrangian/al_solver.jl")
include("solvers/augmented_lagrangian/al_methods.jl")
include("solvers/direct/primals.jl")
include("solvers/direct/pn.jl")
include("solvers/direct/pn_methods.jl")
include("solvers/altro/altro_solver.jl")

include("solvers/direct/moi.jl")
include("solvers/direct/copy_blocks.jl")
include("solvers/direct/direct_constraints.jl")

include("problems.jl")
# include("controllers.jl")

write_ipopt_options()
Logging.global_logger(default_logger(true))

end
