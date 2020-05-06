# export
# 	DynamicsConstraint,
# 	integration

############################################################################################
#                              DYNAMICS CONSTRAINTS										   #
############################################################################################


abstract type AbstractDynamicsConstraint <: CoupledConstraint end
sense(::AbstractDynamicsConstraint) = Equality()
state_dim(con::AbstractDynamicsConstraint) = size(con.model)[1]
control_dim(con::AbstractDynamicsConstraint) = size(con.model)[2]
Base.length(con::AbstractDynamicsConstraint) = size(con.model)[1]

function has_dynamics_constraint(conSet::ConstraintList)
	for con in conSet
		if con isa DynamicsConstraint
			return true
		end
	end
	return false
end

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
struct DynamicsConstraint{Q<:QuadratureRule,L<:AbstractModel,N,M,NM,T} <: AbstractDynamicsConstraint
	model::L
    fVal::Vector{SVector{N,T}}
    xMid::Vector{SVector{N,T}}
    ∇f::Vector{SizedMatrix{N,NM,T,2}}
	A::Vector{SubArray{T,2,SizedMatrix{N,NM,T,2},Tuple{UnitRange{Int},UnitRange{Int}},false}}
	B::Vector{SubArray{T,2,SizedMatrix{N,NM,T,2},Tuple{UnitRange{Int},UnitRange{Int}},false}}
	grad::Vector{GradientExpansion{T,N,M}}
	∇fMid::Vector{SizedMatrix{N,NM,T,2}}
	Am::Vector{SubArray{T,2,SizedMatrix{N,NM,T,2},Tuple{UnitRange{Int},UnitRange{Int}},false}}
	Bm::Vector{SubArray{T,2,SizedMatrix{N,NM,T,2},Tuple{UnitRange{Int},UnitRange{Int}},false}}
end

function DynamicsConstraint{Q}(model::L, N) where {Q,L}
	T = Float64  # TODO: get this from somewhere
	n,m = size(model)
	fVal = [@SVector zeros(n) for k = 1:N]
	xMid = [@SVector zeros(n) for k = 1:N]
	∇f   = [SizedMatrix{n,n+m}(zeros(n,n+m)) for k = 1:N]
	∇fm  = [SizedMatrix{n,n+m}(zeros(n,n+m)) for k = 1:3]
	ix,iu = 1:n, n .+ (1:m)
	A  = [view(∇f[k], ix,ix) for k = 1:N]
	B  = [view(∇f[k], ix,iu) for k = 1:N]
	Am = [view(∇fm[k],ix,ix) for k = 1:3]
	Bm = [view(∇fm[k],ix,iu) for k = 1:3]
	grad  = [GradientExpansion{T}(n,m) for k = 1:3]
	DynamicsConstraint{Q,L,n,m,n+m,T}(model, fVal, xMid, ∇f, A, B,
		grad, ∇fm, Am, Bm)
end

@inline DynamicsConstraint(model, N) = DynamicsConstraint{DEFAULT_Q}(model, N)
integration(::DynamicsConstraint{Q}) where Q = Q

widths(con::DynamicsConstraint{<:Any,<:Any,N,M},n::Int=N,m::Int=M) where {N,M} = (n+m,n+m)
widths(con::DynamicsConstraint{<:Explicit,<:Any,N,M},n::Int=N,m::Int=M) where {N,M} = (n+m,n)
# width(con::DynamicsConstraint{<:Implicit,L,T,N,M,NM}) where {L,T,N,M,NM} = 2N+M
# width(con::DynamicsConstraint{<:Explicit,L,T,N,M,NM}) where {L,T,N,M,NM} = 2NM
####!

# Implicit
function evaluate(con::DynamicsConstraint{Q}, z1::AbstractKnotPoint, z2::AbstractKnotPoint) where Q <: Explicit
	RobotDynamics.discrete_dynamics(Q, con.model, z1) - state(z2)
end

# function evaluate!(vals::Vector{<:AbstractVector}, con::DynamicsConstraint{Q},
# 		Z::Traj, inds=1:length(Z)-1) where Q<:Explicit
# 	for k in inds
# 		vals[k] .= discrete_dynamics(Q, con.model, Z[k]) - state(Z[k+1])
# 	end
# end

# function jacobian!(∇c::Matrix{<:AbstractMatrix}, con::DynamicsConstraint{Q},
# 		Z::Vector{<:AbstractKnotPoint{<:Any,n,m}}, inds=1:length(Z)-1) where {Q<:Explicit,n,m}
# 	In = Diagonal(@SVector ones(n))
# 	ix = Z[1]._x
# 	zinds = [Z[1]._x; Z[1]._u]
# 	for k in inds
# 		# ∇f = uview(∇c[k].data, 1:n, 1:n+m+1)
# 		∇f = ∇c[k,1]
# 		discrete_jacobian!(Q, ∇f, con.model, Z[k])
# 		# ∇f2 = uview(∇c[k].data, 1:n, n+m .+ (1:n))
# 		∇f2 = ∇c[k,2]
# 		∇f2[ix,ix] .= -Diagonal(@SVector ones(n))
# 		return
# 	end
# end

function jacobian!(∇c, con::DynamicsConstraint{Q},
		z::AbstractKnotPoint, z2::AbstractKnotPoint{<:Any,n}, i=1) where {Q,n}
	if i == 1
		RobotDynamics.discrete_jacobian!(Q, ∇c, con.model, z)
		return false  # not constant
	elseif i == 2
		for i = 1:n
			∇c[i,i] = -1
		end
		return true   # is constant
	end
	# return nothing
end
