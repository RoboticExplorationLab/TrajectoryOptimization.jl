
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

# @inline DynamicsConstraint(model, N) = DynamicsConstraint{DEFAULT_Q}(model, N)
# integration(::DynamicsConstraint{Q}) where Q = Q
@inline function widths(con::DynamicsConstraint) 
	n,m = dims(con.model)
	(n+m,n+m)
end

@generated function RD.evaluate!(
	sig::FunctionSignature, 
	con::DynamicsConstraint, 
	vals::Vector{V}, 
	Z::AbstractTrajectory, 
	inds=1:length(Z)-1
) where V
	if sig <: InPlace
		expr = :(RD.dynamics_error!(con.model, vals[i], vals[i+1], Z[k+1], Z[k]))
	elseif sig <: StaticReturn
    	op = V <: SVector ? :(=) : :(.=)
		expr = Expr(op, :(vals[i]), :(RD.dynamics_error(con.model, Z[k+1], Z[k])))
	end
	quote
		for (i, k) in enumerate(inds)	
			$expr
		end
	end
end

function RD.jacobian!(
    sig::FunctionSignature,
    dif::DiffMethod,
    con::DynamicsConstraint,
    ∇c::Matrix{<:AbstractMatrix},
    c::VecOrMat{<:AbstractVector},
    Z::AbstractTrajectory,
    inds = 1:length(Z)-1
)
    for (i, k) in enumerate(inds)
		RD.dynamics_error_jacobian!(sig, dif, con.model, ∇c[i,2], ∇c[i,1], c[i+1], c[i], 
		                            Z[k+1], Z[k])
    end
end

a = 1

# widths(con::DynamicsConstraint{<:Any,<:Any,N,M},n::Int=N,m::Int=M) where {N,M} = (n+m,n+m)
# widths(con::DynamicsConstraint{<:Explicit,<:Any,N,M},n::Int=N,m::Int=M) where {N,M} = (n+m,n)

# get_inds(con::DynamicsConstraint{<:Explicit}, n, m) = (1:n+m, (n+m) .+ (1:n))

# # Explict 
# function evaluate(con::DynamicsConstraint{Q}, z1::AbstractKnotPoint, z2::AbstractKnotPoint) where Q <: Explicit
# 	RobotDynamics.discrete_dynamics(Q, con.model, z1) - state(z2)
# end

# function jacobian!(∇c, con::DynamicsConstraint{Q,L},
# 		z::AbstractKnotPoint, z2::AbstractKnotPoint{<:Any,n}, i=1) where {Q,L,n}
# 	if i == 1
# 		RobotDynamics.discrete_jacobian!(Q, ∇c, con.model, z, con.cache)
# 		return L <: RD.LinearModel
# 	elseif i == 2
# 		for i = 1:n
# 			∇c[i,i] = -1
# 		end
# 		return true   # is constant
# 	end
# 	# return nothing
# end

# function ∇jacobian!(G, con::DynamicsConstraint{<:Explicit},
# 	z::AbstractKnotPoint, z2::AbstractKnotPoint, λ, i=1)
# 	if i == 1
# 		dyn(x) = evaluate(con, StaticKnotPoint(z,x), z2)'λ
# 		G .+= ForwardDiff.hessian(dyn, z.z)
# 	elseif i == 2
# 		nothing
# 	end
# 	return false
# end
