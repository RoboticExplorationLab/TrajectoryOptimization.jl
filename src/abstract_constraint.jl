export
	ConstraintSense,
	Inequality,
	Equality,
	Stage,
	State,
	Control,
	Coupled,
	Dynamical

export
	evaluate,
	jacobian


abstract type GeneralConstraint end

"Specifies whether the constraint is an equality or inequality constraint.
Valid subtypes are `Equality`, `Inequality`, and `Null`"
abstract type ConstraintSense end
"Inequality constraints of the form ``h(x) \\leq 0``"
abstract type Equality <: ConstraintSense end
"Equality constraints of the form ``g(x) = 0``"
abstract type Inequality <: ConstraintSense end
abstract type Null <: ConstraintSense end

"""
Specifies the ``bandedness'' of the constraint. This ends up being the width
of the constraint Jacobian, or the total number of input variables. This is
important to reduce the size of arrays needed to store the Jacobian, as well as
special-case the block matrix algebra.

Current subtypes:
* [`Stage`](@ref)
* [`State`](@ref) <: `Stage`
* [`Control`](@ref) <: `Stage`
* [`Coupled`](@ref)
* [`Dynamical`](@ref) <: `Coupled`
* [`CoupledState`](@ref) <: `Coupled`
* [`CoupledControl`](@ref) <: `Coupled`
* [`General`](@ref)
* [`GeneralState`](@ref) <: `General`
* [`GeneralControl`](@ref) <: `General`
"""
abstract type ConstraintType end
"Only a function of states and controls at a single knotpoint"
abstract type Stage <: ConstraintType end
"Only a function of states at a single knotpoint"
abstract type State <: Stage end
"Only a function of controls at a single knotpoint"
abstract type Control <: Stage end
"Only a function of states and controls at two adjacent knotpoints"
abstract type Coupled <: ConstraintType end
"Only a function of states and two adjacent knotpoints,
and the control at the previous knotpoint, i.e. f(x,u) - x′"
abstract type Dynamical <: Coupled end
"Only a function of states at adjacent knotpoints"
abstract type CoupledState <: Coupled end
"Only a function of controls at adjacent knotpoints"
abstract type CoupledControl <: Coupled end
"A function of all states and controls in the trajectory"
abstract type General <: ConstraintType end
"A function of all states in the trajectory"
abstract type GeneralState <: General end
"A function of all controls in the trajectory"
abstract type GeneralControl <: General end

""" $(TYPEDEF)
Abstract vector-valued constraint of size `P` for a trajectory optimization problem.
May be either inequality or equality (specified by `S<:ConstraintSense`), and be function of
single, adjacent, or all knotpoints (specified by `W<:ConstraintType`).

Interface:
Any constraint type must implement the following interface:
```julia
n = state_dim(::MyCon)
m = control_dim(::MyCon)
p = Base.length(::MyCon)
c = evaluate(::MyCon, args...)   # args determined by W
∇c = jacobian(::MyCon, args...)  # args determined by W
```

The `evaluate` and `jacobian` (identical signatures) methods should have the following signatures
* W <: State: `evaluate(::MyCon, x::SVector)`
* W <: Control: `evaluate(::MyCon, u::SVector)`
* W <: Stage: `evaluate(::MyCon, x, u)`
* W <: Dynamical: `evaluate(::MyCon, x′, x, u)`
* W <: Coupled: `evaluate(::MyCon, x′, u′ x, u)`

Or alternatively,
* W <: Stage: `evaluate(::MyCon, z::KnotPoint)`
* W <: Coupled: `evaluate(::MyCon, z′::KnotPoint, z::KnotPoint)`

The Jacobian method for [`State`](@ref) or [`Control`](@ref) is optional, since it will
	be automatically computed using ForwardDiff. Automatic differentiation
	for other types of constraints is not yet supported.

For W <: State, `control_dim(::MyCon)` doesn't need to be defined. Equivalently, for
	W <: Control, `state_dim(::MyCon)` doesn't need to be defined.

For W <: General, the more general `evaluate` and `jacobian` methods must be used
```julia
evaluate!(vals::Vector{<:AbstractVector}, ::MyCon, Z::Traj, inds=1:length(Z)-1)
jacobian!(∇c::Vector{<:AbstractMatrix}, ::MyCon, Z::Traj, inds=1:length(Z)-1)
```
These methods can be specified for any constraint, instead of the not-in-place functions
	above.
"""
abstract type AbstractConstraint{S<:ConstraintSense,W<:ConstraintType,P} <: GeneralConstraint end

# Getters
"Get type of constraint (bandedness)"
contype(::AbstractConstraint{S,W}) where {S,W} = W
"Get constraint sense (Inequality vs Equality)"
sense(::AbstractConstraint{S}) where S = S

"Returns the width of the constraint Jacobian, i.e. the total number of inputs
to the constraint"
width(con::AbstractConstraint{S,Stage}) where S = state_dim(con) + control_dim(con)
width(con::AbstractConstraint{S,State}) where S = state_dim(con)
width(con::AbstractConstraint{S,Control}) where S = control_dim(con)
width(con::AbstractConstraint{S,Coupled}) where S = 2*(state_dim(con) + control_dim(con))
width(con::AbstractConstraint{S,Dynamical}) where S = 2*state_dim(con) + control_dim(con)
width(con::AbstractConstraint{S,CoupledState}) where S = 2*state_dim(con)
width(con::AbstractConstraint{S,CoupledControl}) where S = 2*control_dim(con)
width(con::AbstractConstraint{S,<:General}) where S = Inf

"Upper bound of the constraint, as a vector, which is 0 for all constraints
(except bound constraints)"
upper_bound(con::AbstractConstraint{Inequality,W,P}) where {P,W} = @SVector zeros(P)
upper_bound(con::AbstractConstraint{Equality,W,P}) where {P,W} = @SVector zeros(P)

"Upper bound of the constraint, as a vector, which is 0 equality and -Inf for inequality
(except bound constraints)"
lower_bound(con::AbstractConstraint{Inequality,W,P}) where {P,W} = -Inf*@SVector ones(P)
lower_bound(con::AbstractConstraint{Equality,W,P}) where {P,W} = @SVector zeros(P)

"Is the constraint a bound constraint or not"
@inline is_bound(con::AbstractConstraint) = false

"Check whether the constraint is consistent with the specified state and control dimensions"
@inline check_dims(con::AbstractConstraint{S,State},n,m) where S = state_dim(con) == n
@inline check_dims(con::AbstractConstraint{S,Control},n,m) where S = control_dim(con) == m
@inline function check_dims(con::AbstractConstraint{S,W},n,m) where {S,W<:ConstraintType}
	state_dim(con) == n && control_dim(con) == m
end

"Size of state vector"
control_dims(::AbstractConstraint{S,State}) where S =
	throw(ErrorException("Cannot get control dimension from a state-only constraint"))

"Size of control vector"
state_dims(::AbstractConstraint{S,Control}) where S =
	throw(ErrorException("Cannot get state dimension from a control-only constraint"))

Base.length(::AbstractConstraint{S,W,P}) where {S,W,P} = P

con_label(::AbstractConstraint, i::Int) = "index $i"

############################################################################################
# 								EVALUATION METHODS 										   #
############################################################################################
"""```
evaluate!(vals::Vector{<:AbstractVector}, con::AbstractConstraint{S,W,P},
	Z, inds=1:length(Z)-1)
```
Evaluate constraints for entire trajectory. This is the most general method used to evaluate
	constraints, and should be the one used in other functions.

For W<:Stage this will loop over calls to `evaluate(con,Z[k])`

For W<:Coupled this will loop over calls to `evaluate(con,Z[k+1],Z[k])`

For W<:General,this must function must be explicitly defined. Other types may define it
	if desired.
"""
function evaluate!(vals::Vector{<:AbstractVector}, con::AbstractConstraint{P,<:Stage},
		Z::Traj, inds=1:length(Z)) where P
	for (i,k) in enumerate(inds)
		vals[i] = evaluate(con, Z[k])
	end
end

function evaluate!(vals::Vector{<:AbstractVector}, con::AbstractConstraint{P,<:Coupled},
		Z::Traj, inds=1:length(Z)-1) where P
	for (i,k) in enumerate(inds)
		vals[i] = evaluate(con, Z[k+1], Z[k])
	end
end

"""```
jacobian!(vals::Vector{<:AbstractVector}, con::AbstractConstraint{S,W,P},
	Z, inds=1:length(Z)-1)
```
Evaluate constraint Jacobians for entire trajectory. This is the most general method used to
	evaluate constraint Jacobians, and should be the one used in other functions.

For W<:Stage this will loop over calls to `jacobian(con,Z[k])`

For W<:Coupled this will loop over calls to `jacobian(con,Z[k+1],Z[k])`

For W<:General,this must function must be explicitly defined. Other types may define it
	if desired.
"""
function jacobian!(∇c::Vector{<:AbstractMatrix}, con::AbstractConstraint{P,<:Stage},
		Z::Traj, inds=1:length(Z)) where P
	for (i,k) in enumerate(inds)
		∇c[i] = jacobian(con, Z[k])
	end
end

function jacobian!(∇c::Vector{<:AbstractMatrix}, con::AbstractConstraint{P,<:Coupled},
	Z::Traj, inds=1:length(Z)-1) where P
	for (i,k) in enumerate(inds)
		∇c[i] = jacobian(con, Z[k+1], Z[k])
	end
end

# Default methods for converting KnotPoints to states and controls
for method in [:evaluate, :jacobian]
	@eval begin
			@inline $(method)(con::AbstractConstraint{S,Stage},   Z::AbstractKnotPoint) where S = $(method)(con, state(Z), control(Z))
			@inline $(method)(con::AbstractConstraint{S,State},   Z::AbstractKnotPoint) where S = $(method)(con, state(Z))
			@inline $(method)(con::AbstractConstraint{S,Control}, Z::AbstractKnotPoint) where S = $(method)(con, control(Z))

			@inline $(method)(con::AbstractConstraint{S,Coupled}, Z′::AbstractKnotPoint, Z::AbstractKnotPoint) where S =
				$(method)(con, state(Z′), control(Z′), state(Z), control(Z))
			@inline $(method)(con::AbstractConstraint{S,Dynamical}, Z′::AbstractKnotPoint, Z::AbstractKnotPoint) where S =
				$(method)(con, state(Z′), state(Z), control(Z))
			@inline $(method)(con::AbstractConstraint{S,CoupledState}, Z′::AbstractKnotPoint, Z::AbstractKnotPoint) where S =
				$(method)(con, state(Z′), state(Z))
			@inline $(method)(con::AbstractConstraint{S,CoupledControl}, Z′::AbstractKnotPoint, Z::AbstractKnotPoint) where S =
				$(method)(con, control(Z′), control(Z))
	end
end

# Method for automatically calculating the gradient for constraints with only 1 input
function jacobian(con::AbstractConstraint{P,W}, x::SVector{N}) where {P,N,W<:Union{State,Control}}
	eval_c(x) = evaluate(con, x)
	ForwardDiff.jacobian(eval_c, x)
end
