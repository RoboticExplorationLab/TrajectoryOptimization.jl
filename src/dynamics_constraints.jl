
############################################################################################
#                              DYNAMICS CONSTRAINTS										   #
############################################################################################


# abstract type AbstractDynamicsConstraint <: CoupledConstraint end

""" $(TYPEDEF)
An equality constraint imposed by the discretized system dynamics. Links adjacent time steps.
Supports both implicit and explicit integration methods. Can store values internally for
more efficient computation of dynamics and dynamics Jacobians over the entire trajectory,
particularly for explicit methods. These constraints are used in Direct solvers, where
the dynamics are explicit stated as constraints in a more general optimization method.

# Constructors
```julia
DynamicsConstraint{Q}(model::AbstractModel, N)
```
where `N` is the number of knot points and `Q<:QuadratureRule` is the integration method.
"""
# struct DynamicsConstraint{L<:DiscreteDynamics,N,M,NM,T} <: AbstractDynamicsConstraint
# 	model::L
#     fVal::Vector{SVector{N,T}}
#     xMid::Vector{SVector{N,T}}
#     ∇f::Vector{SizedMatrix{N,NM,T,2,Matrix{T}}}
# 	A::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}}
# 	B::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}}
# 	grad::Vector{GradientExpansion{T,N,M}}
# 	∇fMid::Vector{SizedMatrix{N,NM,T,2,Matrix{T}}}
# 	Am::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}}
# 	Bm::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}}
# 	cache::FiniteDiff.JacobianCache{Vector{T}, Vector{T}, Vector{T}, 
# 		UnitRange{Int}, Nothing, Val{:forward}(), T}
# end

# function DynamicsConstraint{Q}(model::L, N) where {Q,L}
# 	T = Float64  # TODO: get this from somewhere
# 	n,m = size(model)
# 	fVal = [@SVector zeros(n) for k = 1:N]
# 	xMid = [@SVector zeros(n) for k = 1:N]
# 	∇f   = [SizedMatrix{n,n+m}(zeros(n,n+m)) for k = 1:N]
# 	∇fm  = [SizedMatrix{n,n+m}(zeros(n,n+m)) for k = 1:3]
# 	ix,iu = 1:n, n .+ (1:m)
# 	A  = [view(∇f[k].data, ix,ix) for k = 1:N]
# 	B  = [view(∇f[k].data, ix,iu) for k = 1:N]
# 	Am = [view(∇fm[k].data,ix,ix) for k = 1:3]
# 	Bm = [view(∇fm[k].data,ix,iu) for k = 1:3]
# 	grad  = [GradientExpansion{T}(n,m) for k = 1:3]
# 	cache = FiniteDiff.JacobianCache(model)
# 	DynamicsConstraint{Q,L,n,m,n+m,T}(model, fVal, xMid, ∇f, A, B,
# 		grad, ∇fm, Am, Bm, cache)
# end
struct DynamicsConstraint{L<:DiscreteDynamics} <: CoupledConstraint 
	model::L
	function DynamicsConstraint(model::L) where L <: DiscreteDynamics 
		new{L}(model)
	end
end
function DynamicsConstraint(model::AbstractModel)
	error("A dynamics constraint can only be constructed from a discrete dynamics model")
end

@inline sense(::DynamicsConstraint) = Equality()
@inline state_dim(con::DynamicsConstraint) = state_dim(con.model)
@inline control_dim(con::DynamicsConstraint) = control_dim(con.model)
@inline Base.length(con::DynamicsConstraint) = state_dim(con.model)
RD.output_dim(con::DynamicsConstraint) = state_dim(con.model)
RD.default_diffmethod(con::DynamicsConstraint) = RD.default_diffmethod(con.model)

# @inline DynamicsConstraint(model, N) = DynamicsConstraint{DEFAULT_Q}(model, N)
# integration(::DynamicsConstraint{Q}) where Q = Q
@inline function widths(con::DynamicsConstraint) 
	n,m = dims(con.model)
	(n+m,n+m)
end

function evaluate_constraints!(
	sig::InPlace, 
	con::DynamicsConstraint, 
	vals::Vector{V}, 
	Z::SampledTrajectory, 
	inds=1:length(Z)-1
) where V
	for (i, k) in enumerate(inds)	
		RD.dynamics_error!(con.model, vals[i], vals[i+1], Z[k+1], Z[k])
	end
end

@generated function evaluate_constraints!(
	sig::StaticReturn, 
	con::DynamicsConstraint, 
	vals::Vector{V}, 
	Z::SampledTrajectory, 
	inds=1:length(Z)-1
) where V
	op = V <: SVector ? :(=) : :(.=)
	expr = Expr(op, :(vals[i]), :(RD.dynamics_error(con.model, Z[k+1], Z[k])))
	quote
		for (i, k) in enumerate(inds)	
			$expr
		end
	end
end

function constraint_jacobians!(
    sig::FunctionSignature,
    dif::DiffMethod,
    con::DynamicsConstraint,
    ∇c::Matrix{<:AbstractMatrix},
    c::VecOrMat{<:AbstractVector},
    Z::SampledTrajectory,
    inds = 1:length(Z)-1
)
    for (i, k) in enumerate(inds)
		RD.dynamics_error_jacobian!(sig, dif, con.model, ∇c[i,2], ∇c[i,1], c[i+1], c[i], 
		                            Z[k+1], Z[k])
    end
end