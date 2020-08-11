"""
    TrajectoryOptimization
Primary module for setting up and evaluating trajectory optimization problems.
"""
module TrajectoryOptimization

# using Rotations
using StaticArrays
using LinearAlgebra
using DocStringExtensions
using ForwardDiff
using UnsafeArrays
using SparseArrays
using MathOptInterface
const MOI = MathOptInterface

import RobotDynamics

using RobotDynamics: AbstractModel, LieGroupModel,
	KnotPoint, StaticKnotPoint, AbstractKnotPoint,
	QuadratureRule, Implicit, Explicit, DEFAULT_Q, HermiteSimpson,
	is_terminal, state_diff, state_diff_jacobian!, state_diff_jacobian,
	state, control, states, controls, get_times, Traj, AbstractTrajectory

import RobotDynamics: jacobian!, state_dim, control_dim, states, controls, 
	state_diff_jacobian!, rollout!

# API
export  # types
	Problem,
	Objective,
	LQRObjective,
	ConstraintList,
	DiagonalCost,
	QuadraticCost,
	LQRCost,
	Traj,
	TrajOptNLP,
	KnotPoint   # from RobotDynamics

export  # methods
	cost,
	max_violation,
	initial_controls!,
	initial_states!,
	initial_trajectory!,
	rollout!,
	states,
	controls,
	get_trajectory,
	state_dim,    # from RobotDynamics
	control_dim   # from RobotDynamics

export
	Equality,
	Inequality,
	BoundConstraint,
	CircleConstraint,
	SphereConstraint,
	GoalConstraint,
	LinearConstraint,
	CollisionConstraint,
	NormConstraint,
	add_constraint!

include("expansions.jl")
include("costfunctions.jl")
include("objective.jl")

include("abstract_constraint.jl")
include("constraints.jl")
include("dynamics_constraints.jl")
include("constraint_list.jl")
include("integration.jl")

include("cost.jl")
include("convals.jl")

include("problem.jl")
include("conset.jl")
include("ALconset.jl")

include("nlp.jl")

include("utils.jl")
end
