import RobotDynamics: jacobian!

"""
    AbstractConstraint <: RobotDynamics.AbstractFunction

Abstract vector-valued constraint for a trajectory optimization problem.
Defined using the `AbstractFunction` interface from RobotDynamics.jl which 
allows for a flexible API for using in-place or out-of-place evaluation, 
multiple methods for evaluating Jacobians, etc.

The "sense" of a constraint is specified by defining the [`ConstraintSense`](@ref)
trait on the type, accessed via `TrajectoryOptimization.sense`, which defines 
whether the constraint is an equality, inequality, or conic constraint.

Interface:
Any constraint must implement the following interface:
```julia
n = RobotDynamics.state_dim(::MyCon)
m = RobotDynamics.control_dim(::MyCon)
p = RobotDynamics.output_dim(::MyCon)
TrajectoryOptimization.sense(::MyCon)::ConstraintSense
c = RobotDynamics.evaluate(::MyCon, x, u)
RobotDynamics.evaluate!(::MyCon, c, x, u)
```

Constraints 

All constraints are categorized into the following type tree:
```text
          AbstractConstraint
                  ↓
           StageConstraint
            ↙            ↘ 
StageConstraint       ControlConstraint
```

The state and control dimensions (where applicable) can be queried using
`state_dim(::AbstractConstraint)` and `control_dim(::AbstractConstraint)`.
The dimensions of a constraint can be verified using [`check_dims`](@ref).

The width of the Jacobian is specified by whether the constraint inherits 
from `StageConstraint`, `StateConstraint`, or `ControlConstraint`. It can 
be generated automatically using [`gen_jacobian`](@ref).

## Evaluation
All constraints can be evaluated by using one of these methods using a 
`KnotPoint`:

    RobotDynamics.evaluate(::MyCon, z::AbstractKnotPoint)
    RobotDynamics.evaluate!(::MyCon, c, z::AbstractKnotPoint)

Alternatively, a `StageConstraint` can be evaluated using

    RobotDynamics.evaluate(::MyCon, x, u)
    RobotDynamics.evaluate!(::MyCon, c, x, u)

a `StateConstraint` can be evaluated using

    RobotDynamics.evaluate(::MyCon, x)
    RobotDynamics.evaluate!(::MyCon, c, x)

and a `ControlConstraint can be evaluated using`

    RobotDynamics.evaluate(::MyCon, u)
    RobotDynamics.evaluate!(::MyCon, c, u)

## Jacobian Evaluation
The Jacobian for all types of constraints can be evaluated using 

    RobotDynamics.jacobian!(::FunctionSignature, ::DiffMethod, ::MyCon, J, c, z)

where `z` is an `AbstractKnotPoint`. To define custom Jacobians, one of the following
methods must be defined:

    RobotDynamics.jacobian!(::MyCon, J, c, z)     # All constraints 
    RobotDynamics.jacobian!(::MyCon, J, c, x, u)  # StageConstraint only
    RobotDynamics.jacobian!(::MyCon, J, c, x)     # StateConstraint only
    RobotDynamics.jacobian!(::MyCon, J, c, u)     # ControlConstraint only

The most specific methods are preferred over those that accept only an `AbstractKnotPoint`.
"""
abstract type AbstractConstraint <: RD.AbstractFunction end

"Only a function of states and controls at a single knotpoint"
abstract type StageConstraint <: AbstractConstraint end
RD.functioninputs(::CollisionConstraint) = RD.StateControl()

"Only a function of states at a single knotpoint"
abstract type StateConstraint <: StageConstraint end
RD.functioninputs(::StateConstraint) = RD.StateOnly()

"Only a function of controls at a single knotpoint"
abstract type ControlConstraint <: StageConstraint end
RD.functioninputs(::ControlConstraint) = RD.ControlOnly()

"Get constraint sense (Inequality vs Equality)"
sense(::C) where {C<:AbstractConstraint} = throw(NotImplemented(:sense, Symbol(C)))

Base.copy(con::AbstractConstraint) = con

""" 
    upper_bound(constraint)

Upper bound of the constraint, as a vector. This is zero for inequality and equality
constraints, and +Inf for [`SecondOrderCone`](@ref).
"""
@inline upper_bound(con::AbstractConstraint) =
    upper_bound(sense(con)) * @SVector ones(RD.output_dim(con))
@inline upper_bound(::Inequality) = 0.0
@inline upper_bound(::Equality) = 0.0
@inline upper_bound(::SecondOrderCone) = Inf

""" 
    lower_bound(constraint)

Lower bound of the constraint, as a vector. This is zero for equality constraints
and -Inf for [`SecondOrderCone`](@ref) and inequality constraints.
"""
@inline lower_bound(con::AbstractConstraint) =
    lower_bound(sense(con)) * @SVector ones(RD.output_dim(con))
@inline lower_bound(::Inequality) = -Inf
@inline lower_bound(::Equality) = 0.0
@inline lower_bound(::SecondOrderCone) = -Inf

"""
    is_bound(constraints)

Returns true if the constraint can be represeted as either

```math
    x_\\text{min} \\leq x \\leq x_\\text{max}
```
or 
```math
    u_\\text{min} \\leq u \\leq u_\\text{max}
```
i.e. simple bound constraints on the states and controls.
"""
@inline is_bound(con::AbstractConstraint) = false

"Check whether the constraint is consistent with the specified state and control dimensions"
@inline check_dims(con::StateConstraint, n, m) = state_dim(con) == n
@inline check_dims(con::ControlConstraint, n, m) = control_dim(con) == m
@inline check_dims(con::AbstractConstraint, n, m) =
    state_dim(con) == n && control_dim(con) == m

con_label(::AbstractConstraint, i::Int) = "index $i"

"""
    constraintlabel(con, i)

Return a string describing the `i`th index of `con`. Default is simply `"index \$i"`.
"""
constraintlabel(::AbstractConstraint, i::Integer) = "index $i"


"""
    gen_jacobian(con)

Generate a Jacobian of the correct size for constraint `con`.
"""
gen_jacobian(con::AbstractConstraint) = gen_jacobian(Float64, con)
function gen_jacobian(::Type{T}, con::AbstractConstraint) where T
    nm = RD.input_dim(con)
    p = RD.output_dim(con)
    zeros(T, p, nm)
end


############################################################################################
# 								EVALUATION METHODS 										   #
############################################################################################
function evaluate_constraint!(::StaticReturn, con::AbstractConstraint, val, args...)
	val .= RD.evaluate(con, args...)
end

function evaluate_constraint!(::InPlace, con::AbstractConstraint, val, args...)
	RD.evaluate!(con, val, args...)
    val
end

function constraint_jacobian!(sig::FunctionSignature, diff::DiffMethod, con, jac, val, args...)
    RD.jacobian!(sig, diff, con, jac, val, args...)
end


"""
    evaluate_constraints!(sig, con, vals, Z, inds)

Evaluate the constraint `con` using the `sig` `FunctionSignature` for the time steps in 
`inds` along trajectory `Z`, storing the output in `vals`.

The `vals` argument should be a vector with the same length as `inds`, where each element 
is a mutable vector of length `RD.output_dim(con)`.
"""
@generated function evaluate_constraints!(
    sig::StaticReturn,
    con::StageConstraint,
    vals::Vector{V},
    Z::SampledTrajectory,
    inds = 1:length(Z)
) where V
    op = V <: SVector ? :(=) : :(.=)
    quote
        for (i, k) in enumerate(inds)
            $(Expr(op, :(vals[i]), :(RD.evaluate(con, Z[k]))))
        end
    end
end

function evaluate_constraints!(
    sig::InPlace,
    con::StageConstraint,
    vals::Vector{<:AbstractVector},
    Z::SampledTrajectory,
    inds = 1:length(Z)
)
    for (i, k) in enumerate(inds)
        RD.evaluate!(con, vals[i], Z[k])
    end
end

"""
    constraint_jacobians!(sig, diffmethod, con, vals, Z, inds)

Evaluate the constraint `con` using the `sig` `FunctionSignature` for the time steps in 
`inds` along trajectory `Z`, storing the output in `vals`.

The `vals` argument should be a vector with the same length as `inds`, where each element 
is a mutable vector of length `RD.output_dim(con)`.
"""
function constraint_jacobians!(
    sig::FunctionSignature,
    dif::DiffMethod,
    con::StageConstraint,
    ∇c::VecOrMat{<:AbstractMatrix},
    c::VecOrMat{<:AbstractVector},
    Z::SampledTrajectory,
    inds = 1:length(Z)
)
    for (i, k) in enumerate(inds)
        RD.jacobian!(sig, dif, con, ∇c[i], c[i], Z[k])
    end
end


"""
    ∇constraint_jacobians!(G, con::AbstractConstraint, Z, λ, inds, is_const, init)
    ∇constraint_jacobians!(G, con::AbstractConstraint, Z::AbstractKnotPoint, λ::AbstractVector)

Evaluate the second-order expansion of the constraint `con` along the trajectory `Z`
after multiplying by the lagrange multiplier `λ`.
The optional `is_const` argument is a `BitArray` of the same size as `∇c`, and captures 
the output of `jacobian!`, which should return a Boolean specifying if the Jacobian is
constant or not. The `init` flag will force re-calculation of constant Jacobians when true.

The method for each constraint should calculate the Jacobian of the vector-Jacobian product,
    and therefore should be of size n × n if the input dimension is n.

Importantly, this method should ADD and not overwrite the contents of `G`, since this term
is dependent upon all the constraints acting at that time step.
"""
function ∇constraint_jacobians!(
    sig::FunctionSignature,
    dif::DiffMethod,
    con::StageConstraint,
    H::VecOrMat{<:AbstractMatrix},
    λ::VecOrMat{<:AbstractVector},
    c::VecOrMat{<:AbstractVector},
    Z::SampledTrajectory,
    inds = 1:length(Z)
)
    for (i, k) in enumerate(inds)
        ∇jacobian!(con, H[i], λ[i], c[i], Z[k])
    end
end

function error_expansion!(jac, jac0, con::StageConstraint, model::DiscreteDynamics, G, inds) where C
	if jac !== jac0
		n,m = RD.dims(model)
        n̄ = RD.errstate_dim(model)
		ix = 1:n̄
		iu = n̄ .+ (1:m)
		ix0 = 1:n
		iu0 = n .+ (1:m)
		for (i,k) in enumerate(inds)
            ∇x  = view(jac[i], :, ix)
            ∇u  = view(jac[i], :, iu)
            ∇x0 = view(jac0[i], :, ix0)
            ∇u0 = view(jac0[i], :, iu0)

			if con isa StateConstraints
				mul!(∇x, ∇x0, get_data(G[k]))
			elseif con isa ControlConstraints
				∇u .= ∇u0
			end
		end
	end
end
