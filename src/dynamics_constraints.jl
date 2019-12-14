export
	DynamicsConstraint

############################################################################################
#                              DYNAMICS CONSTRAINTS										   #
############################################################################################


abstract type AbstractDynamicsConstraint{W<:Coupled,P} <: AbstractConstraint{Equality,W,P} end
state_dim(con::AbstractDynamicsConstraint) = size(con.model)[1]
control_dim(con::AbstractDynamicsConstraint) = size(con.model)[2]
Base.length(con::AbstractDynamicsConstraint) = size(con.model)[1]


struct DynamicsConstraint{Q<:QuadratureRule,L<:AbstractModel,T,N,M,NM} <: AbstractDynamicsConstraint{Coupled,N}
	model::L
    fVal::Vector{SVector{N,T}}
    xMid::Vector{SVector{N,T}}
    ∇f::Vector{SMatrix{N,M,T,NM}}
end

function DynamicsConstraint{Q}(model::L, N) where {Q,L}
	T = Float64  # TODO: get this from somewhere
	n,m = size(model)
	fVal = [@SVector zeros(n) for k = 1:N]
	xMid = [@SVector zeros(n) for k = 1:N]
	∇f = [@SMatrix zeros(n,n+m) for k = 1:N]
	DynamicsConstraint{Q,L,T,n,n+m,(n+m)n}(model, fVal, xMid, ∇f)
end

@inline DynamicsConstraint(model, N) = DynamicsConstraint{DEFAULT_Q}(model, N)
integration(::DynamicsConstraint{Q}) where Q = Q

width(con::DynamicsConstraint{<:Implicit,L,T,N,NM}) where {L,T,N,NM} = 2N+NM-N
width(con::DynamicsConstraint{<:Explicit,L,T,N,NM}) where {L,T,N,NM} = 2NM


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



struct DynamicsVals{T,N,M,L}
    fVal::Vector{SVector{N,T}}
    xMid::Vector{SVector{N,T}}
    ∇f::Vector{SMatrix{N,M,T,L}}
end

function DynamicsVals(dyn_con::DynamicsConstraint)
	DynamicsVals(dyn_con.fVal, dyn_con.xMid, dyn_con.∇f)
end
