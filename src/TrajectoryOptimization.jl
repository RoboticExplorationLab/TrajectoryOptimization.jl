module TrajectoryOptimization

# using Rotations
using StaticArrays
using LinearAlgebra
using DocStringExtensions
using ForwardDiff
using FiniteDiff
using UnsafeArrays
using SparseArrays
using MathOptInterface
using Rotations
const MOI = MathOptInterface

import RobotDynamics
const RD = RobotDynamics

using RobotDynamics: AbstractModel, DiscreteDynamics, LieGroupModel, DiscreteLieDynamics,
	KnotPoint, StaticKnotPoint, AbstractKnotPoint,
	QuadratureRule, Implicit, Explicit, 
	state_dim, control_dim, output_dim,
	is_terminal, state_diff, state_diff_jacobian!,
	state, control, states, controls, gettimes, SampledTrajectory,
	num_vars, dims,
	FunctionSignature, DiffMethod,
	FiniteDifference, ForwardAD, StaticReturn, InPlace, UserDefined

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
	get_times,
	get_objective,
	get_constraints,
	get_model,
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

include("costfunctions.jl")
include("quadratic_costs.jl")
include("lie_costs.jl")
include("objective.jl")

include("cones.jl")
include("abstract_constraint.jl")
include("constraints.jl")
include("constraint_list.jl")

include("cost.jl")

include("problem.jl")


include("utils.jl")

import Base.length
@deprecate length(con::AbstractConstraint) RobotDynamics.output_dim(con)
end
