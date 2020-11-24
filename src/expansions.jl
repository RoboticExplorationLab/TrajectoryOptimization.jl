abstract type AbstractExpansion{T} end

struct GradientExpansion{T,N,M} <: AbstractExpansion{T}
	x::SizedVector{N,T,1}
	u::SizedVector{M,T,1}
	function GradientExpansion{T}(n::Int,m::Int) where T
		new{T,n,m}(SizedVector{n}(zeros(T,n)), SizedVector{m}(zeros(T,m)))
	end
end

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

function save_tmp!(D::DynamicsExpansion)
	D.tmpA .= D.A_
	D.tmpB .= D.B_
end

function dynamics_expansion!(Q, D::Vector{<:DynamicsExpansion}, model::AbstractModel,
		Z::Traj)
	for k in eachindex(D)
		RobotDynamics.discrete_jacobian!(Q, D[k].∇f, model, Z[k])
		# save_tmp!(D[k])
		# D[k].tmpA .= D[k].A_  # avoids allocations later
		# D[k].tmpB .= D[k].B_
	end
end

# function dynamics_expansion!(D::Vector{<:DynamicsExpansion}, model::AbstractModel,
# 		Z::Traj, Q=RobotDynamics.RK3)
# 	for k in eachindex(D)
# 		RobotDynamics.discrete_jacobian!(Q, D[k].∇f, model, Z[k])
# 		D[k].tmpA .= D[k].A_  # avoids allocations later
# 		D[k].tmpB .= D[k].B_
# 	end
# end


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

function error_expansion!(D::Vector{<:DynamicsExpansion}, model::AbstractModel, G)
	for d in D
		save_tmp!(d)
	end
end

function error_expansion!(D::Vector{<:DynamicsExpansion}, model::LieGroupModel, G)
	for k in eachindex(D)
		save_tmp!(D[k])
		error_expansion!(D[k], G[k], G[k+1])
	end
end

# function linearize(::Type{Q}, model::AbstractModel, z::AbstractKnotPoint) where Q
# 	D = DynamicsExpansion(model)
# 	linearize!(Q, D, model, z)
# end
#
# function linearize!(::Type{Q}, D::DynamicsExpansion{<:Any,<:Any,N,M}, model::AbstractModel,
# 		z::AbstractKnotPoint) where {N,M,Q}
# 	discrete_jacobian!(Q, D.∇f, model, z)
# 	D.tmpA .= D.A_  # avoids allocations later
# 	D.tmpB .= D.B_
# 	return D.tmpA, D.tmpB
# end
#
# function linearize!(::Type{Q}, D::DynamicsExpansion, model::LieGroupModel) where Q
# 	discrete_jacobian!(Q, D.∇f, model, z)
# 	D.tmpA .= D.A_  # avoids allocations later
# 	D.tmpB .= D.B_
# 	G1 = state_diff_jacobian(model, state(z))
# 	G2 = state_diff_jacobian(model, x1)
# 	error_expansion!(D, G1, G2)
# 	return D.A, D.B
# end
