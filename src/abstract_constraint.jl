import RobotDynamics: jacobian!


"Specifies whether the constraint is an equality or inequality constraint.
Valid subtypes are `Equality`, `Inequality`, and `Null`"
abstract type ConstraintSense end
"Inequality constraints of the form ``h(x) \\leq 0``"
struct Equality <: ConstraintSense end
"Equality constraints of the form ``g(x) = 0``"
struct Inequality <: ConstraintSense end

"""
	AbstractConstraint

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
evaluate!(vals::Vector{<:AbstractVector}, ::MyCon, Z::AbstractTrajectory, inds=1:length(Z)-1)
jacobian!(∇c::Vector{<:AbstractMatrix}, ::MyCon, Z::AbstractTrajectory, inds=1:length(Z)-1)
```
These methods can be specified for any constraint, instead of the not-in-place functions
	above.
"""
abstract type AbstractConstraint end

"Only a function of states and controls at a single knotpoint"
abstract type StageConstraint <: AbstractConstraint end
"Only a function of states at a single knotpoint"
abstract type StateConstraint <: StageConstraint end
"Only a function of controls at a single knotpoint"
abstract type ControlConstraint <: StageConstraint end
"Only a function of states and controls at two adjacent knotpoints"
abstract type CoupledConstraint <: AbstractConstraint end
"Only a function of states at adjacent knotpoints"
abstract type CoupledStateConstraint <: CoupledConstraint end
"Only a function of controls at adjacent knotpoints"
abstract type CoupledControlConstraint <: CoupledConstraint end

const StateConstraints = Union{StageConstraint, StateConstraint, CoupledConstraint, CoupledStateConstraint}
const ControlConstraints = Union{StageConstraint, ControlConstraint, CoupledConstraint, CoupledControlConstraint}

"Get constraint sense (Inequality vs Equality)"
sense(::C) where C <: AbstractConstraint = throw(NotImplemented(:sense, Symbol(C)))

"Get type of constraint (bandedness)"
contype(::C) where C <: AbstractConstraint = throw(NotImplemented(:contype, Symbol(C)))

"Dimension of the state vector"
RobotDynamics.state_dim(::C) where C <: StateConstraint = throw(NotImplemented(:state_dim, Symbol(C)))

"Dimension of the control vector"
RobotDynamics.control_dim(::C) where C <: ControlConstraint = throw(NotImplemented(:control_dim, Symbol(C)))

"Return the constraint value"
evaluate(::C) where C <: AbstractConstraint = throw(NotImplemented(:evaluate, Symbol(C)))

"Length of constraint vector"
Base.length(::C) where C <: AbstractConstraint = throw(NotImplemented(:length, Symbol(C)))

# widths(con::StageConstraint, n=state_dim(con), m=control_dim(con)) = (n+m,)
# widths(con::StateConstraint, n=state_dim(con), m=0) = (n,)
# widths(con::ControlConstraint, n=0, m=control_dim(con)) = (m,)
# widths(con::CoupledConstraint, n=state_dim(con), m=control_dim(con)) = (n+m, n+m)
# widths(con::CoupledStateConstraint, n=state_dim(con), m=0) = (n,n)
# widths(con::CoupledControlConstraint, n=0, m=control_dim(con)) = (m,m)

"Upper bound of the constraint, as a vector, which is 0 for all constraints
(except bound constraints)"
@inline upper_bound(con::AbstractConstraint) = upper_bound(sense(con)) * @SVector ones(length(con))
@inline upper_bound(::Inequality) = 0.0
@inline upper_bound(::Equality) = 0.0

"Upper bound of the constraint, as a vector, which is 0 equality and -Inf for inequality
(except bound constraints)"
@inline lower_bound(con::AbstractConstraint) = lower_bound(sense(con)) * @SVector ones(length(con))
@inline lower_bound(::Inequality) = -Inf
@inline lower_bound(::Equality) = 0.0

"""
	primal_bounds!(zL, zU, con::AbstractConstraint)

Set the lower `zL` and upper `zU` bounds on the primal variables imposed by the constraint
`con`. Return whether or not the vectors `zL` or `zU` could be modified by `con`
(i.e. if the constraint `con` is a bound constraint).
"""
primal_bounds!(zL, zU, con::AbstractConstraint) = false

"Is the constraint a bound constraint or not"
@inline is_bound(con::AbstractConstraint) = false

"Check whether the constraint is consistent with the specified state and control dimensions"
@inline check_dims(con::StateConstraint,n,m) = state_dim(con) == n
@inline check_dims(con::ControlConstraint,n,m) = control_dim(con) == m
@inline check_dims(con::AbstractConstraint,n,m) = state_dim(con) == n && control_dim(con) == m

get_dims(con::Union{StateConstraint,CoupledStateConstraint}, nm::Int) =
	state_dim(con), nm - state_dim(con)
get_dims(con::Union{ControlConstraint,CoupledControlConstraint}, nm::Int) =
	nm - control_dim(con), control_dim(con)
get_dims(con::AbstractConstraint, nm::Int) = state_dim(con), control_dim(con)

con_label(::AbstractConstraint, i::Int) = "index $i"

"""
	get_inds(con::AbstractConstraint)

Get the indices of the joint state-control vector that are used to calculate the constraint.
If the constraint depends on more than one time step, the indices start from the beginning
of the first one.
"""
get_inds(con::StateConstraint, n, m) = (1:n,)
get_inds(con::ControlConstraint, n, m) = (n .+ (1:m),)
get_inds(con::StageConstraint, n, m) = (1:n+m,)
get_inds(con::CoupledConstraint, n, m) = (1:n+m, n+m+1:2n+2m)

@inline widths(con::AbstractConstraint, n, m) = length.(get_inds(con, n, m))
@inline widths(con::StageConstraint) = (state_dim(con)+control_dim(con),)
@inline widths(con::StateConstraint) = (state_dim(con),)
@inline widths(con::ControlConstraint) = (control_dim(con),)
@inline widths(con::CoupledConstraint) = (state_dim(con)+control_dim(con),state_dim(con)+control_dim(con))
@inline widths(con::CoupledStateConstraint) = (state_dim(con),state_dim(con))
@inline widths(con::CoupledControlConstraint) = (control_dim(con),control_dim(con))

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
function evaluate!(vals::Vector{<:AbstractVector}, con::StageConstraint,
		Z::AbstractTrajectory, inds=1:length(Z))
	for (i,k) in enumerate(inds)
		vals[i] .= evaluate(con, Z[k])
	end
end

function evaluate!(vals::Vector{<:AbstractVector}, con::CoupledConstraint,
		Z::AbstractTrajectory, inds=1:length(Z)-1)
	for (i,k) in enumerate(inds)
		vals[i] .= evaluate(con, Z[k], Z[k+1])
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
function jacobian!(∇c::VecOrMat{<:AbstractMatrix}, con::StageConstraint,
		Z::AbstractTrajectory, inds=1:length(Z))
	for (i,k) in enumerate(inds)
		jacobian!(∇c[i], con, Z[k])
	end
end

function jacobian!(∇c::VecOrMat{<:AbstractMatrix}, con::CoupledConstraint,
		Z::AbstractTrajectory, inds=1:size(∇c,1))
	for (i,k) in enumerate(inds)
		jacobian!(∇c[i,1], con, Z[k], Z[k+1], 1)
		jacobian!(∇c[i,2], con, Z[k], Z[k+1], 2)
	end
end

"""
    ∇jacobian!(G, con::AbstractConstraint, Z, λ, inds, is_const, init)
    ∇jacobian!(G, con::AbstractConstraint, Z::AbstractKnotPoint, λ::AbstractVector)

Evaluate the second-order expansion of the constraint `con` along the trajectory `Z`
after multiplying by the lagrange multiplier `λ`.

The method for each constraint should calculate the Jacobian of the vector-Jacobian product,
    and therefore should be of size n × n if the input dimension is n.

Importantly, this method should ADD and not overwrite the contents of `G`, since this term
is dependent upon all the constraints acting at that time step.
"""
function ∇jacobian!(G::VecOrMat{<:AbstractMatrix}, con::StageConstraint,
		Z::AbstractTrajectory, λ::Vector{<:AbstractVector},
		inds=1:length(Z), is_const=ones(Bool,length(inds)), init::Bool=false)
	for (i,k) in enumerate(inds)
		if init || !is_const[i]
			is_const[i] = ∇jacobian!(G[i], con, Z[k], λ[i])
		end
	end
end

function ∇jacobian!(G::VecOrMat{<:AbstractMatrix}, con::CoupledConstraint,
		Z::AbstractTrajectory, λ::Vector{<:AbstractVector},
		inds=1:length(Z), is_const=ones(Bool,length(inds)), init::Bool=false)
	for (i,k) in enumerate(inds)
		if init || !is_const[i]
			is_const[i] = ∇jacobian!(G[i,1], con, Z[k], Z[k+1], λ[i], 1)
			is_const[i] = ∇jacobian!(G[i,2], con, Z[k], Z[k+1], λ[i], 2)
		end
	end
end

# Default methods for converting KnotPoints to states and controls for StageConstraints
@inline evaluate(con::StateConstraint, z::AbstractKnotPoint) = evaluate(con, state(z))
@inline evaluate(con::ControlConstraint, z::AbstractKnotPoint) = evaluate(con, control(z))
@inline evaluate(con::StageConstraint, z::AbstractKnotPoint) = evaluate(con, state(z), control(z))


"""
	jacobian!(∇c::AbstractMatrix, con::StageConstraint, z::AbstractKnotPoint)
	jacobian!(∇c::AbstractMatrix, con::StageConstraint, x::StaticVector, u::StaticVector)
	jacobian!(∇c::AbstractMatrix, con::StateConstraint, x::StaticVector)
	jacobian!(∇c::AbstractMatrix, con::ControlConstraint, u::StaticVector)

Evaluate the constraint Jacobian for a `StageConstraint`. Any `StageConstraint` must implement
	one of these methods.
"""
jacobian!(∇c, con::StateConstraint, z::AbstractKnotPoint, i=1) =
	jacobian!(∇c, con, state(z))
jacobian!(∇c, con::ControlConstraint, z::AbstractKnotPoint, i=1) =
	jacobian!(∇c, con, control(z))
jacobian!(∇c, con::StageConstraint, z::AbstractKnotPoint, i=1) =
	jacobian!(∇c, con, state(z), control(z))

# ForwardDiff jacobians that are of only state or control
function jacobian!(∇c, con::StageConstraint, x::StaticVector)
	eval_c(x) = evaluate(con, x)
	∇c .= ForwardDiff.jacobian(eval_c, x)
	return false
end

@inline ∇jacobian!(G, con::StateConstraint, z::AbstractKnotPoint, λ, i=1) =
	∇jacobian!(G, con, state(z), λ)
@inline ∇jacobian!(G, con::ControlConstraint, z::AbstractKnotPoint, λ, i=1) =
	∇jacobian!(G, con, control(z), λ)
@inline ∇jacobian!(G, con::StageConstraint, z::AbstractKnotPoint, λ, i=1) =
	∇jacobian!(G, con, state(z), control(z), λ)

function ∇jacobian!(G, con::StageConstraint, x::StaticVector, λ)
	eval_c(x) = evaluate(con, x)'λ
	G_ = ForwardDiff.hessian(eval_c, x)
    G .+= G_
	return false
end

function ∇jacobian!(G, con::StageConstraint, x::StaticVector{n}, u::StaticVector{m}, λ) where {n,m}
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    eval_c(z) = evaluate(con, z[ix], z[iu])'λ
    G .+= ForwardDiff.hessian(eval_c, [x;u])
    return false
end


function gen_jacobian(con::AbstractConstraint,i=1)
	ws = widths(con)
	p = length(con)
	C1 = SizedMatrix{p,ws[i]}(zeros(p,ws[i]))
end

function gen_views(∇c::AbstractMatrix, con::StateConstraint, n=state_dim(con), m=0)
	view(∇c,:,1:n), view(∇c,:,n:n-1)
end

function gen_views(∇c::AbstractMatrix, con::ControlConstraint, n=0, m=control_dim(con))
	view(∇c,:,1:0), view(∇c,:,1:m)
end

function gen_views(∇c::AbstractMatrix, con::AbstractConstraint, n=state_dim(con), m=control_dim(con))
	if size(∇c,2) < n+m
		view(∇c,:,1:n), view(∇c,:,n:n-1)
	else
		view(∇c,:,1:n), view(∇c,:,n .+ (1:m))
	end
end
