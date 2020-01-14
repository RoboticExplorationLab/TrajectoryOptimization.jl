export
	DynamicsConstraint,
	integration

############################################################################################
#                              DYNAMICS CONSTRAINTS										   #
############################################################################################


abstract type AbstractDynamicsConstraint{W<:Coupled,P} <: AbstractConstraint{Equality,W,P} end
state_dim(con::AbstractDynamicsConstraint) = size(con.model)[1]
control_dim(con::AbstractDynamicsConstraint) = size(con.model)[2]
Base.length(con::AbstractDynamicsConstraint) = size(con.model)[1]

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
struct DynamicsConstraint{Q<:QuadratureRule,L<:AbstractModel,T,N,W,A} <: AbstractDynamicsConstraint{Coupled,N}
	model::L
    fVal::Vector{SVector{N,T}}
    xMid::Vector{SVector{N,T}}
    ∇f::Vector{A}
end

function DynamicsConstraint{Q}(model::L, N) where {Q,L}
	T = Float64  # TODO: get this from somewhere
	n,m = size(model)
	fVal = [@SVector zeros(n) for k = 1:N]
	xMid = [@SVector zeros(n) for k = 1:N]
	if n*(n+m) > MAX_ELEM
		∇f = [zeros(n,n+m) for k = 1:N]
	else
		∇f = [@SMatrix zeros(n,n+m) for k = 1:N]
	end
	NM = n+m
	DynamicsConstraint{Q,L,T,n,NM,eltype(∇f)}(model, fVal, xMid, ∇f)
end

@inline DynamicsConstraint(model, N) = DynamicsConstraint{DEFAULT_Q}(model, N)
integration(::DynamicsConstraint{Q}) where Q = Q

width(con::DynamicsConstraint{<:Implicit,L,T,N,NM}) where {L,T,N,NM} = 2N+NM-N
width(con::DynamicsConstraint{<:Explicit,L,T,N,NM}) where {L,T,N,NM} = 2NM
####!

# Implicit
function evaluate!(vals::Vector{<:AbstractVector}, con::DynamicsConstraint{Q},
		Z::Traj, inds=1:length(Z)-1) where Q<:Implicit
	for k in inds
		vals[k] = discrete_dynamics(Q, con.model, Z[k]) - state(Z[k+1])
	end
end

function jacobian!(∇c::Vector{<:AbstractMatrix}, con::DynamicsConstraint{Q,L,T,N},
		Z::Traj, inds=1:length(Z)-1) where {Q<:Implicit,L,T,N}
	In = Diagonal(@SVector ones(N))
	zinds = [Z[1]._x; Z[1]._u]
	for k in inds
		AB = discrete_jacobian(Q, con.model, Z[k])
		∇c[k] = [AB[:,zinds] -In]
	end
end



struct DynamicsVals{T,N,A}
    fVal::Vector{SVector{N,T}}
    xMid::Vector{SVector{N,T}}
    ∇f::Vector{A}
end

function DynamicsVals(dyn_con::DynamicsConstraint)
	DynamicsVals(dyn_con.fVal, dyn_con.xMid, dyn_con.∇f)
end
