import RobotDynamics: jacobian!

"""
    AbstractConstraint

Abstract vector-valued constraint for a trajectory optimization problem.
May be either inequality or equality (specified by `sense(::AbstractConstraint)::ConstraintSense`),
and be function of single or adjacent knotpoints.

Interface:
Any constraint type must implement the following interface:
```julia
n = state_dim(::MyCon)
m = control_dim(::MyCon)
p = Base.length(::MyCon)
sense(::MyCon)::ConstraintSense
c = evaluate(::MyCon, args...)
jacobian!(∇c, ::MyCon, args...)
```

All constraints are categorized into the following type tree:
```text
                        AbstractConstraint
                        ↙                ↘
           StageConstraint               CoupledConstraint
            ↙        ↘                       ↙           ↘
StageConstraint ControlConstraint CoupledStateConstraint CoupledControlConstraint
```

The state and control dimensions (where applicable) can be queried using
`state_dim(::AbstractConstraint)` and `control_dim(::AbstractConstraint)`.
The dimensions of a constraint can be verified using [`check_dims`](@ref).
The width of the constraint Jacobian is given by [`get_inds`](@ref) or [`widths`](@ref).

The number of constraint values associated with the constraint (length of the constraint vector)
is given with `length(::AbstractConstraint)`.

# Evaluation methods
Refer to the doc strings for the following methods for more information on the required
signatures.
* [`evaluate`](@ref)
* [`jacobian!`](@ref)
* [`∇jacobian!`](@ref)
"""
abstract type AbstractConstraint <: RD.AbstractFunction end

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

const StateConstraints =
    Union{StageConstraint,StateConstraint,CoupledConstraint,CoupledStateConstraint}
const ControlConstraints =
    Union{StageConstraint,ControlConstraint,CoupledConstraint,CoupledControlConstraint}

"Get constraint sense (Inequality vs Equality)"
sense(::C) where {C<:AbstractConstraint} = throw(NotImplemented(:sense, Symbol(C)))

"Dimension of the state vector"
RobotDynamics.state_dim(::C) where {C<:StateConstraint} =
    throw(NotImplemented(:state_dim, Symbol(C)))

"Dimension of the control vector"
RobotDynamics.control_dim(::C) where {C<:ControlConstraint} =
    throw(NotImplemented(:control_dim, Symbol(C)))

"Return the constraint value"
evaluate(::C) where {C<:AbstractConstraint} = throw(NotImplemented(:evaluate, Symbol(C)))

"Length of constraint vector (deprecated for RD.output_dim)"
# Base.length(con::C) where {C<:AbstractConstraint} = RD.output_dim(con) 
RD.output_dim(::C) where {C<:AbstractConstraint} = throw(NotImplemented(:output_dim, Symbol(C)))

# widths(con::StageConstraint, n=state_dim(con), m=control_dim(con)) = (n+m,)
# widths(con::StateConstraint, n=state_dim(con), m=0) = (n,)
# widths(con::ControlConstraint, n=0, m=control_dim(con)) = (m,)
# widths(con::CoupledConstraint, n=state_dim(con), m=control_dim(con)) = (n+m, n+m)
# widths(con::CoupledStateConstraint, n=state_dim(con), m=0) = (n,n)
# widths(con::CoupledControlConstraint, n=0, m=control_dim(con)) = (m,m)

"Upper bound of the constraint, as a vector, which is 0 for all constraints
(except bound constraints)"
@inline upper_bound(con::AbstractConstraint) =
    upper_bound(sense(con)) * @SVector ones(RD.output_dim(con))
@inline upper_bound(::Inequality) = 0.0
@inline upper_bound(::Equality) = 0.0

"Upper bound of the constraint, as a vector, which is 0 equality and -Inf for inequality
(except bound constraints)"
@inline lower_bound(con::AbstractConstraint) =
    lower_bound(sense(con)) * @SVector ones(RD.output_dim(con))
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
@inline check_dims(con::StateConstraint, n, m) = state_dim(con) == n
@inline check_dims(con::ControlConstraint, n, m) = control_dim(con) == m
@inline check_dims(con::AbstractConstraint, n, m) =
    state_dim(con) == n && control_dim(con) == m

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

"""
    widths(::AbstractConstraint)
    widths(::AbstractConstraint, n, m)

Return a tuple of the widths of the Jacobians for a constraint. If `n` and `m` are not passed
in, they are assumed to be consistent with those returned by `state_dim` and `control_dim`.
"""
@inline widths(con::AbstractConstraint, n, m) = length.(get_inds(con, n, m))
@inline widths(con::StageConstraint) = (state_dim(con) + control_dim(con),)
@inline widths(con::StateConstraint) = (state_dim(con),)
@inline widths(con::ControlConstraint) = (control_dim(con),)
@inline widths(con::CoupledConstraint) =
    (state_dim(con) + control_dim(con), state_dim(con) + control_dim(con))
@inline widths(con::CoupledStateConstraint) = (state_dim(con), state_dim(con))
@inline widths(con::CoupledControlConstraint) = (control_dim(con), control_dim(con))

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
    evaluate!(vals, con::AbstractConstraint, Z, [inds])

Evaluate constraints for entire trajectory. This is the most general method used to evaluate
constraints along the trajectory `Z`, and should be the one used in other functions.
The `inds` argument determines at which knot points the constraint is evaluated.

If `con` is a `StageConstraint`, this will call `evaluate(con, z)` by default, or
`evaluate(con, z1, z2)` if `con` is a `CoupledConstraint`.
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

# function evaluate!(
#     vals::Vector{<:AbstractVector},
#     con::StageConstraint,
#     Z::SampledTrajectory,
#     inds = 1:length(Z),
# )
#     for (i, k) in enumerate(inds)
#         vals[i] .= evaluate(con, Z[k])
#     end
# end

# function evaluate!(
#     vals::Vector{<:AbstractVector},
#     con::CoupledConstraint,
#     Z::SampledTrajectory,
#     inds = 1:length(Z)-1,
# )
#     for (i, k) in enumerate(inds)
#         vals[i] .= evaluate(con, Z[k], Z[k+1])
#     end
# end

"""
    jacobian!(∇c, con::AbstractConstraint, Z, [inds, is_const])

Evaluate constraints for entire trajectory. This is the most general method used to evaluate
constraints along the trajectory `Z`, and should be the one used in other functions.
The `inds` argument determines at which knot points the constraint is evaluated.
The optional `is_const` argument is a `BitArray` of the same size as `∇c`, and captures 
the output of `jacobian!`, which should return a Boolean specifying if the Jacobian is
constant or not.

The values are stored in `∇c`, which should be a matrix of matrices. If `con` is a
`StageConstraint`, `size(∇c,2) = 1`, and `size(∇c,2) = 2` if `con` is a `CoupledConstraint`.

If `con` is a `StageConstraint`, this will call `jacobian!(∇c, con, z)` by default, or
`jacobian!(∇c, con, z1, z2, i)` if `con` is a `CoupledConstraint`.
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

# function jacobian!(
#     ∇c::VecOrMat{<:AbstractMatrix},
#     con::StageConstraint,
#     Z::SampledTrajectory,
#     inds = 1:length(Z),
#     is_const = BitArray(undef, size(∇c))
# )
#     for (i, k) in enumerate(inds)
#         is_const[i] = jacobian!(∇c[i], con, Z[k])
#     end
# end

# function jacobian!(
#     ∇c::VecOrMat{<:AbstractMatrix},
#     con::CoupledConstraint,
#     Z::SampledTrajectory,
#     inds = 1:size(∇c, 1),
#     is_const = BitArray(undef, size(∇c))
# )
#     for (i, k) in enumerate(inds)
#         is_const[i,1] = jacobian!(∇c[i, 1], con, Z[k], Z[k+1], 1)
#         is_const[i,2] = jacobian!(∇c[i, 2], con, Z[k], Z[k+1], 2)
#     end
# end

"""
    ∇jacobian!(G, con::AbstractConstraint, Z, λ, inds, is_const, init)
    ∇jacobian!(G, con::AbstractConstraint, Z::AbstractKnotPoint, λ::AbstractVector)

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
function RD.∇jacobian!(
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

# function ∇jacobian!(
#     G::VecOrMat{<:AbstractMatrix},
#     con::StageConstraint,
#     Z::SampledTrajectory,
#     λ::Vector{<:AbstractVector},
#     inds = 1:length(Z),
#     is_const = ones(Bool, length(inds)),
#     init::Bool = false,
# )
#     for (i, k) in enumerate(inds)
#         if init || !is_const[i]
#             is_const[i] = ∇jacobian!(G[i], con, Z[k], λ[i])
#         end
#     end
# end

# function ∇jacobian!(
#     G::VecOrMat{<:AbstractMatrix},
#     con::CoupledConstraint,
#     Z::SampledTrajectory,
#     λ::Vector{<:AbstractVector},
#     inds = 1:length(Z),
#     is_const = ones(Bool, length(inds)),
#     init::Bool = false,
# )
#     for (i, k) in enumerate(inds)
#         if init || !is_const[i]
#             is_const[i] = ∇jacobian!(G[i, 1], con, Z[k], Z[k+1], λ[i], 1)
#             is_const[i] = ∇jacobian!(G[i, 2], con, Z[k], Z[k+1], λ[i], 2)
#         end
#     end
# end

# # Default methods for converting KnotPoints to states and controls for StageConstraints
# """
#     evaluate(con::AbstractConstraint, z)       # stage constraint
#     evaluate(con::AbstractConstraint, z1, z2)  # coupled constraint

# Evaluate the constraint `con` at knot point `z`. By default, this method will attempt to call

#     evaluate(con, x)

# if `con` is a `StateConstraint`,

#     evaluate(con, u)

# if `con` is a `ControlConstraint`, or

#     evaluate(con, x, u)

# if `con` is a `StageConstraint`. If `con` is a `CoupledConstraint` the constraint should
# define

#     evaluate(con, z1, z2)

# """
# @inline evaluate(con::StateConstraint, z::AbstractKnotPoint) = evaluate(con, state(z))
# @inline evaluate(con::ControlConstraint, z::AbstractKnotPoint) = evaluate(con, control(z))
# @inline evaluate(con::StageConstraint, z::AbstractKnotPoint) =
#     evaluate(con, state(z), control(z))


# """
#     jacobian!(∇c, con::AbstractConstraint, z, i=1)       # stage constraint
#     jacobian!(∇c, con::AbstractConstraint, z1, z2, i=1)  # coupled constraint

# Evaluate the constraint `con` at knot point `z`. By default, this method will attempt to call

#     jacobian!(∇c, con, x)

# if `con` is a `StateConstraint`,

#     jacobian!(∇c, con, u)

# if `con` is a `ControlConstraint`, or

#     jacobian!(∇c, con, x, u)

# if `con` is a `StageConstraint`. If `con` is a `CoupledConstraint` the constraint should
# define

#     jacobian!(∇c, con, z, i)

# where `i` determines which Jacobian should be evaluated. E.g. if `i = 1`, the Jacobian
# with respect to the first knot point's stage and controls is calculated.

# # Automatic Differentiation
# If `con` is a `StateConstraint` or `ControlConstraint` then this method is automatically
# defined using ForwardDiff.
# """
# jacobian!(∇c, con::StateConstraint, z::AbstractKnotPoint, i = 1) =
#     jacobian!(∇c, con, state(z))
# jacobian!(∇c, con::ControlConstraint, z::AbstractKnotPoint, i = 1) =
#     jacobian!(∇c, con, control(z))
# jacobian!(∇c, con::StageConstraint, z::AbstractKnotPoint, i = 1) =
#     jacobian!(∇c, con, state(z), control(z))

# # ForwardDiff jacobians that are of only state or control
# function jacobian!(∇c, con::StageConstraint, x::StaticVector)
#     eval_c(x) = evaluate(con, x)
#     ∇c .= ForwardDiff.jacobian(eval_c, x)
#     return false
# end

# @inline ∇jacobian!(G, con::StateConstraint, z::AbstractKnotPoint, λ, i = 1) =
#     ∇jacobian!(G, con, state(z), λ)
# @inline ∇jacobian!(G, con::ControlConstraint, z::AbstractKnotPoint, λ, i = 1) =
#     ∇jacobian!(G, con, control(z), λ)
# @inline ∇jacobian!(G, con::StageConstraint, z::AbstractKnotPoint, λ, i = 1) =
#     ∇jacobian!(G, con, state(z), control(z), λ)

# function ∇jacobian!(G, con::StageConstraint, x::StaticVector, λ)
#     eval_c(x) = evaluate(con, x)'λ
#     G_ = ForwardDiff.hessian(eval_c, x)
#     G .+= G_
#     return false
# end

# function ∇jacobian!(
#     G,
#     con::StageConstraint,
#     x::StaticVector{n},
#     u::StaticVector{m},
#     λ,
# ) where {n,m}
#     ix = SVector{n}(1:n)
#     iu = SVector{m}(n .+ (1:m))
#     eval_c(z) = evaluate(con, z[ix], z[iu])'λ
#     G .+= ForwardDiff.hessian(eval_c, [x; u])
#     return false
# end


function gen_jacobian(con::AbstractConstraint, i = 1)
    ws = widths(con)
    p = RD.output_dim(con)
    C1 = SizedMatrix{p,ws[i]}(zeros(p, ws[i]))
end

# function gen_views(∇c::AbstractMatrix, con::StateConstraint, n = state_dim(con), m = 0)
#     view(∇c, :, 1:n), view(∇c, :, n:n-1)
# end

# function gen_views(∇c::AbstractMatrix, con::ControlConstraint, n = 0, m = control_dim(con))
#     view(∇c, :, 1:0), view(∇c, :, 1:m)
# end

# function gen_views(
#     ∇c::AbstractMatrix,
#     con::AbstractConstraint,
#     n = state_dim(con),
#     m = control_dim(con),
# )
#     if size(∇c, 2) < n + m
#         view(∇c, :, 1:n), view(∇c, :, n:n-1)
#     else
#         view(∇c, :, 1:n), view(∇c, :, n .+ (1:m))
#     end
# end