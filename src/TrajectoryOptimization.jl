"""
    TrajectoryOptimization
Primary module for setting up and solving trajectory optimization problems.
"""
module TrajectoryOptimization

# using Rotations
using StaticArrays
using LinearAlgebra
using DocStringExtensions
using ForwardDiff
using UnsafeArrays
import RobotDynamics

using RobotDynamics: AbstractModel, LieGroupModel,
	KnotPoint, StaticKnotPoint, AbstractKnotPoint,
	QuadratureRule, Implicit, Explicit, DEFAULT_Q, HermiteSimpson,
	is_terminal, state_diff, state_diff_jacobian!, state_diff_jacobian,
	state, control

import RobotDynamics: jacobian!, state_dim, control_dim, state_diff_jacobian!

# API
export  # types
	Problem,
	Objective,
	LQRObjective,
	ConstraintList,
	DiagonalCost,
	QuadraticCost,
	LQRCost,
	Traj

export  # methods
	cost,
	max_violation,
	initial_controls!,
	initial_states!,
	initial_trajectory!,
	rollout!

export
	BoundConstraint,
	CircleConstraint,
	SphereConstraint,
	GoalConstraint,
	add_constraint!

include("trajectories.jl")
include("expansions.jl")
include("costfunctions.jl")
include("objective.jl")

include("abstract_constraint.jl")
include("constraints.jl")
include("dynamics_constraints.jl")
include("integration.jl")

include("cost.jl")
include("convals.jl")

include("problem.jl")
include("conset.jl")

include("utils.jl")
end
