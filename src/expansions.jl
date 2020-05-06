abstract type AbstractExpansion{T} end

struct GradientExpansion{T,N,M} <: AbstractExpansion{T}
	x::SizedVector{N,T,1}
	u::SizedVector{M,T,1}
	function GradientExpansion{T}(n::Int,m::Int) where T
		new{T,n,m}(SizedVector{n}(zeros(T,n)), SizedVector{m}(zeros(T,m)))
	end
end
#
# struct CostExpansion{T,N0,N,M} <: AbstractExpansion{T}
# 	# Cost Expansion Terms
# 	x ::SizedVector{N0,T,1}
# 	xx::SizedMatrix{N0,N0,T,2}
# 	u ::SizedVector{M,T,1}
# 	uu::SizedMatrix{M,M,T,2}
# 	ux::SizedMatrix{M,N0,T,2}
#
# 	# Error Expansion Terms
# 	x_ ::SizedVector{N,T,1}
# 	xx_::SizedMatrix{N,N,T,2}
# 	u_ ::SizedVector{M,T,1}
# 	uu_::SizedMatrix{M,M,T,2}
# 	ux_::SizedMatrix{M,N,T,2}
#
# 	tmp::SizedMatrix{N0,N,T,2}
# 	x0::SizedVector{N0,T,1}  # gradient of cost function only (no multipliers)
# end
#
# function CostExpansion{T}(n0::Int, n::Int, m::Int) where T
# 	x0  = SizedVector{n0}(zeros(T,n0))
# 	xx0 = SizedMatrix{n0,n0}(zeros(T,n0,n0))
# 	u0  = SizedVector{m}(zeros(T,m))
# 	uu0 = SizedMatrix{m,m}(zeros(T,m,m))
# 	ux0 = SizedMatrix{m,n0}(zeros(T,m,n0))
#
# 	x  = SizedVector{n}(zeros(T,n))
# 	xx = SizedMatrix{n,n}(zeros(T,n,n))
# 	u  = SizedVector{m}(zeros(T,m))
# 	uu = SizedMatrix{m,m}(zeros(T,m,m))
# 	ux = SizedMatrix{m,n}(zeros(T,m,n))
# 	tmp = SizedMatrix{n0,n}(zeros(T,n0,n))
# 	x_ = copy(x0)
# 	CostExpansion(x0,xx0,u0,uu0,ux0, x, xx, u, uu, ux, tmp, x_)
# end
#
# function CostExpansion{T}(n::Int, m::Int) where T
# 	x  = SizedVector{n}(zeros(T,n))
# 	xx = SizedMatrix{n,n}(zeros(T,n,n))
# 	u  = SizedVector{m}(zeros(T,m))
# 	uu = SizedMatrix{m,m}(zeros(T,m,m))
# 	ux = SizedMatrix{m,n}(zeros(T,m,n))
# 	tmp = SizedMatrix{n,n}(zeros(T,n,n))
# 	x_ = copy(x)
# 	CostExpansion(x,xx,u,uu,ux, x, xx, u, uu, ux, tmp, x_)
# end

#
# struct Expansion{T,N0,N,M} <: AbstractExpansion{T}
# 	x::SizedVector{N,T,1}
# 	xx::SizedMatrix{N,N,T,2}
# 	u::SizedVector{M,T,1}
# 	uu::SizedMatrix{M,M,T,2}
# 	ux::SizedMatrix{M,N,T,2}
# 	tmp::SizedMatrix{N0,N,T,2}
# 	function Expansion{T}(n::Int) where T
# 		x = SizedVector{n}(zeros(n))
# 		xx = SizedMatrix{n,n}(zeros(n,n))
# 		new{T,n,n,0}(x,xx)
# 	end
# 	function Expansion{T}(n::Int,m::Int) where T
# 		x = SizedVector{n}(zeros(n))
# 		xx = SizedMatrix{n,n}(zeros(n,n))
# 		u = SizedVector{m}(zeros(m))
# 		uu = SizedMatrix{m,m}(zeros(m,m))
# 		ux = SizedMatrix{m,n}(zeros(m,n))
# 		new{T,n,n,m}(x,xx,u,uu,ux)
# 	end
# 	function Expansion{T}(n0::Int,n::Int,m::Int) where T
# 		x = SizedVector{n}(zeros(n))
# 		xx = SizedMatrix{n,n}(zeros(n,n))
# 		u = SizedVector{m}(zeros(m))
# 		uu = SizedMatrix{m,m}(zeros(m,m))
# 		ux = SizedMatrix{m,n}(zeros(m,n))
# 		tmp = SizedMatrix{n0,n}(zeros(n0,n))
# 		new{T,n0,n,m}(x,xx,u,uu,ux,tmp)
# 	end
# 	function Expansion(
# 			x::SizedVector{N,T,1},
# 			xx::SizedMatrix{N,N,T,2},
# 			u::SizedVector{M,T,1},
# 			uu::SizedMatrix{M,M,T,2},
# 			ux::SizedMatrix{M,N,T,2}) where {T,N,M}
# 		new{T,N,N,M}(x,xx,u,uu,ux)
# 	end
# end

# TODO: Move to ALTRO
struct DynamicsExpansion{T,N,N̄,M}
	∇f::Matrix{T} # n × (n+m+1)
	A_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
	B_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
	A::SizedMatrix{N̄,N̄,T,2}
	B::SizedMatrix{N̄,M,T,2}
	tmpA::SizedMatrix{N,N,T,2}
	tmpB::SizedMatrix{N,M,T,2}
	tmp::SizedMatrix{N,N̄,T,2}
	function DynamicsExpansion{T}(n0::Int, n::Int, m::Int) where T
		∇f = zeros(n0,n0+m)
		ix = 1:n0
		iu = n0 .+ (1:m)
		A_ = view(∇f, ix, ix)
		B_ = view(∇f, ix, iu)
		A = SizedMatrix{n,n}(zeros(n,n))
		B = SizedMatrix{n,m}(zeros(n,m))
		tmpA = SizedMatrix{n0,n0}(zeros(n0,n0))
		tmpB = SizedMatrix{n0,m}(zeros(n0,m))
		tmp = zeros(n0,n)
		new{T,n0,n,m}(∇f,A_,B_,A,B,tmpA,tmpB,tmp)
	end
	function DynamicsExpansion{T}(n::Int, m::Int) where T
		∇f = zeros(n,n+m)
		ix = 1:n
		iu = n .+ (1:m)
		A_ = view(∇f, ix, ix)
		B_ = view(∇f, ix, iu)
		A = SizedMatrix{n,n}(zeros(n,n))
		B = SizedMatrix{n,m}(zeros(n,m))
		tmpA = A
		tmpB = B
		tmp = zeros(n,n)
		new{T,n,n,m}(∇f,A_,B_,A,B,tmpA,tmpB,tmp)
	end
end

function dynamics_expansion!(Q, D::Vector{<:DynamicsExpansion}, model::AbstractModel,
		Z::Traj)
	for k in eachindex(D)
		RobotDynamics.discrete_jacobian!(Q, D[k].∇f, model, Z[k])
		D[k].tmpA .= D[k].A_  # avoids allocations later
		D[k].tmpB .= D[k].B_
	end
end

function linearize(::Type{Q}, model::AbstractModel, z::AbstractKnotPoint) where Q
	D = DynamicsExpansion(model)
	linearize!(Q, D, model, z)
end

function linearize!(::Type{Q}, D::DynamicsExpansion{<:Any,<:Any,N,M}, model::AbstractModel,
		z::AbstractKnotPoint) where {N,M,Q}
	discrete_jacobian!(Q, D.∇f, model, z)
	D.tmpA .= D.A_  # avoids allocations later
	D.tmpB .= D.B_
	return D.tmpA, D.tmpB
end

function linearize!(::Type{Q}, D::DynamicsExpansion, model::LieGroupModel) where Q
	discrete_jacobian!(Q, D.∇f, model, z)
	D.tmpA .= D.A_  # avoids allocations later
	D.tmpB .= D.B_
	G1 = state_diff_jacobian(model, state(z))
	G2 = state_diff_jacobian(model, x1)
	error_expansion!(D, G1, G2)
	return D.A, D.B
end

function error_expansion!(D::DynamicsExpansion,G1,G2)
    mul!(D.tmp, D.tmpA, G1)
    mul!(D.A, Transpose(G2), D.tmp)
    mul!(D.B, Transpose(G2), D.tmpB)
end

@inline error_expansion(D::DynamicsExpansion, model::LieGroupModel) = D.A, D.B
@inline error_expansion(D::DynamicsExpansion, model::AbstractModel) = D.tmpA, D.tmpB
@inline DynamicsExpansion(model::AbstractModel) = DynamicsExpansion{Float64}(model)
@inline function DynamicsExpansion{T}(model::AbstractModel) where T
	n,m = size(model)
	n̄ = state_diff_size(model)
	DynamicsExpansion{T}(n,n̄,m)
end

@inline error_expansion!(D::Vector{<:DynamicsExpansion}, model::AbstractModel, G) = nothing

function error_expansion!(D::Vector{<:DynamicsExpansion}, model::LieGroupModel, G)
	for k in eachindex(D)
		error_expansion!(D[k], G[k], G[k+1])
	end
end
