abstract type AbstractExpansion{T} end

struct Expansion{T} <: AbstractExpansion{T}
	x::Vector{T}
	xx::Matrix{T}
	u::Vector{T}
	uu::Matrix{T}
	ux::Matrix{T}
	tmp::Matrix{T}
	function Expansion{T}(n::Int) where T
		x = zeros(n)
		xx = zeros(n,n)
		new{T}(x,xx)
	end
	function Expansion{T}(n::Int,m::Int) where T
		x = zeros(n)
		xx = zeros(n,n)
		u = zeros(m)
		uu = zeros(m,m)
		ux = zeros(m,n)
		new{T}(x,xx,u,uu,ux)
	end
	function Expansion{T}(n0::Int,n::Int,m::Int) where T
		x = zeros(n)
		xx = zeros(n,n)
		u = zeros(m)
		uu = zeros(m,m)
		ux = zeros(m,n)
		tmp = zeros(n0,n)
		new{T}(x,xx,u,uu,ux,tmp)
	end
end

struct SizedExpansion{T,N,N̄,M} <: AbstractExpansion{T}
	x::SizedVector{N,T,1}
	xx::SizedMatrix{N,N,T,2}
	u::SizedVector{M,T,1}
	uu::SizedMatrix{M,M,T,2}
	ux::SizedMatrix{M,N,T,2}
	tmp::SizedMatrix{N,N̄,T,2}
	function SizedExpansion{T}(n::Int) where T
		x = SizedVector{n}(zeros(n))
		xx = SizedMatrix{n,n}(zeros(n,n))
		new{T,n,n,0}(x,xx)
	end
	function SizedExpansion{T}(n::Int,m::Int) where T
		x = SizedVector{n}(zeros(n))
		xx = SizedMatrix{n,n}(zeros(n,n))
		u = SizedVector{m}(zeros(m))
		uu = SizedMatrix{m,m}(zeros(m,m))
		ux = SizedMatrix{m,n}(zeros(m,n))
		new{T,n,n,m}(x,xx,u,uu,ux)
	end
	function SizedExpansion{T}(n0::Int,n::Int,m::Int) where T
		x = SizedVector{n}(zeros(n))
		xx = SizedMatrix{n,n}(zeros(n,n))
		u = SizedVector{m}(zeros(m))
		uu = SizedMatrix{m,m}(zeros(m,m))
		ux = SizedMatrix{m,n}(zeros(m,n))
		tmp = SizedMatrix{n0,n}(zeros(n0,n))
		new{T,n,n,m}(x,xx,u,uu,ux,tmp)
	end
end


struct GeneralExpansion{T,X,XX,U,UU,UX,TMP} <: AbstractExpansion{T}
	x::X
	xx::XX
	u::U
	uu::UU
	ux::UX
	tmp::TMP
	function GeneralExpansion(x::X, xx::XX, u::U, uu::UU, ux::UX, tmp::TMP) where {X,XX,U,UU,UX,TMP}
		T = eltype(X)
		new{T,X,XX,U,UU,UX,TMP}(x, xx, u, uu, ux, tmp)
	end
end

function GeneralExpansion{T}(n0::Int,n::Int,m::Int) where T
	x   = zeros(T,n)
	xx  = zeros(T,n,n)
	u   = zeros(T,m)
	uu  = zeros(T,m,m)
	ux  = zeros(T,m,n)
	tmp = zeros(T,n0,n)
	GeneralExpansion(x,xx,u,uu,ux,tmp)
end

function GeneralExpansion{T}(::Type{<:MArray}, n0::Int, n::Int, m::Int) where T
	x   = @MVector zeros(T,n)
	xx  = @MMatrix zeros(T,n,n)
	u   = @MVector zeros(T,m)
	uu  = @MMatrix zeros(T,m,m)
	ux  = @MMatrix zeros(T,m,n)
	tmp = @MMatrix zeros(T,n0,n)
	GeneralExpansion(x,xx,u,uu,ux,tmp)
end

function GeneralExpansion{T}(::Type{<:SizedArray}, n0::Int, n::Int, m::Int) where T
	x   = SizedVector{n}(zeros(T,n))
	xx  = SizedMatrix{n,n}(zeros(T,n,n))
	u   = SizedVector{m}(zeros(T,m))
	uu  = SizedMatrix{m,m}(zeros(T,m,m))
	ux  = SizedMatrix{m,n}(zeros(T,m,n))
	tmp = SizedMatrix{n0,n}(zeros(T,n0,n))
	GeneralExpansion(x,xx,u,uu,ux,tmp)
end

function Base.copyto!(E1::AbstractExpansion, E2::AbstractExpansion)
	E1.x .= E2.x
	E1.xx .= E2.xx
	E1.u .= E2.u
	E1.uu .= E2.uu
	E1.ux .= E2.ux
	return nothing
end

struct DynamicsExpansion{T} <: AbstractExpansion{T}
	∇f::Matrix{T} # n × (n+m+1)
	A_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
	B_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
	A::Matrix{T} # nbar × nbar
	B::Matrix{T} # nbar × m
	tmp::Matrix{T} # n × nbar
	function DynamicsExpansion{T}(n0::Int, n::Int, m::Int) where T
		∇f = zeros(n,n+m+1)
		ix = 1:n
		iu = n .+ (1:m)
		A_ = view(∇f, ix, ix)
		B_ = view(∇f, ix, iu)
		A = zeros(n,n)
		B = zeros(n,m)
		tmp = zeros(n0,n)
		new{T}(∇f,A_,B_,A,B,tmp)
	end
end

struct SizedDynamicsExpansion{T,N,N̄,M} <: AbstractExpansion{T}
	∇f::Matrix{T} # n × (n+m+1)
	A_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
	B_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
	A::SizedMatrix{N̄,N̄,T,2}
	B::SizedMatrix{N̄,M,T,2}
	tmpA::SizedMatrix{N,N,T,2}
	tmpB::SizedMatrix{N,M,T,2}
	tmp::SizedMatrix{N,N̄,T,2}
	function SizedDynamicsExpansion{T}(n0::Int, n::Int, m::Int) where T
		∇f = zeros(n,n+m+1)
		ix = 1:n
		iu = n .+ (1:m)
		A_ = view(∇f, ix, ix)
		B_ = view(∇f, ix, iu)
		A = SizedMatrix{n,n}(zeros(n,n))
		B = SizedMatrix{n,m}(zeros(n,m))
		tmpA = SizedMatrix{n0,n0}(zeros(n0,n0))
		tmpB = SizedMatrix{n0,m}(zeros(n0,m))
		tmp = zeros(n0,n)
		new{T,n0,n,m}(∇f,A_,B_,A,B,tmpA,tmpB,tmp)
	end
end
