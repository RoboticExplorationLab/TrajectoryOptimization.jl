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
	state, control, states, controls, gettimes, Traj, AbstractTrajectory,
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

include("expansions.jl")
include("costfunctions.jl")
include("quadratic_costs.jl")
include("lie_costs.jl")
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

# include("nlp.jl")

include("utils.jl")
# include("deprecated.jl")

import Base.length
@deprecate length(con::AbstractConstraint) RobotDynamics.output_dim(con)
end
