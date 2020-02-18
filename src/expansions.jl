
struct Expansion{T}
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

struct DynamicsExpansion{T}
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
