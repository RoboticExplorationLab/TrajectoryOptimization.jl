using StaticArrays, ForwardDiff, BenchmarkTools, LinearAlgebra

abstract type ConstraintType end
abstract type Equality <: ConstraintType end
abstract type Inequality <: ConstraintType end
abstract type Null <: ConstraintType end
abstract type AbstractConstraint{S<:ConstraintType} end


struct KnotConstraint{T,P,NM,PNM,C}
	con::C
	inds::UnitRange{Int}
	vals::Vector{SVector{P,T}}
	vals_prev::Vector{SVector{P,T}}
	∇c::Vector{SMatrix{P,NM,T,PNM}}
	λ::Vector{SVector{P,T}}
	μ::Vector{SVector{P,T}}
	active::Vector{SVector{P,Bool}}
	c_max::Vector{T}
end

function KnotConstraint(con::C, inds::UnitRange) where C
	n,m,p = size(con)
	P = length(inds)
	λ    = [@SVector zeros(p) for k = 1:P]
	μ    = [@SVector ones(p)  for k = 1:P]
	atv  = [@SVector ones(Bool,p) for k = 1:P]
	vals = [@SVector zeros(p) for k = 1:P]
	∇c   = [@SMatrix zeros(p,n+m) for k = 1:P]
	KnotConstraint(con, inds, vals, deepcopy(vals), ∇c, λ, μ, atv, zeros(P))
end

function _index(con::KnotConstraint, k::Int)
	if k ∈ con.inds
		return k - con.inds[1] + 1
	else
		return 0
	end
end

Base.length(::KnotConstraint{T,P,NM,PNM,C}) where {T,P,NM,PNM,C} = P
Base.length(con::KnotConstraint, k::Int) = k ∈ con.inds ? length(con) : 0
Base.size(con::KnotConstraint) = size(con.con)
duals(con::KnotConstraint) = con.λ
duals(con::KnotConstraint, k::Int) = con.λ[_index(con,k)]
penalty(con::KnotConstraint) = con.μ
penalty(con::KnotConstraint, k::Int) = con.μ[_index(con,k)]

function evaluate(con::KnotConstraint, Z::Traj)
	for i in eachindex(con.vals)
		k = con.inds[i]
		con.vals[i] = evaluate(con.con, state(Z[k]), control(Z[k]))
	end
end

function jacobian(con::KnotConstraint, Z::Traj)
	for i in eachindex(con.vals)
		k = con.inds[i]
		con.∇c[i] = jacobian(con.con, Z[k])
	end
end

function update_active_set!(con::KnotConstraint{T,P,NM,PNM,C}, tol=0.0) where
		{T,P,NM,PNM,C<:AbstractConstraint{Inequality}}
	for i in eachindex(con.vals)
		con.active[i] = @. (con.vals[i] >= tol) | (con.λ[i] > 0)
	end
	return nothing
end

update_active_set!(con::KnotConstraint{T,P,NM,PNM,C}, tol=0.0) where
	{T,P,NM,PNM,C<:AbstractConstraint{Equality}} = nothing

function max_violation!(con::KnotConstraint{T,P,NM,PNM,C}) where
		{T,P,NM,PNM,C<:AbstractConstraint{Inequality}}
	for i in eachindex(con.c_max)
		con.c_max[i] = maximum(pos.(con.vals[i]))
	end
	return nothing
end

function max_violation!(con::KnotConstraint{T,P,NM,PNM,C}) where
		{T,P,NM,PNM,C<:AbstractConstraint{Equality}}
	for i in eachindex(con.c_max)
		con.c_max[i] = norm(con.vals[i],Inf)
	end
	return nothing
end




struct ConstraintSet{T}
	constraints::Vector{<:KnotConstraint}
	p::Vector{T}
	c_max::Vector{T}
end

Base.length(conSet::ConstraintSet, k) = constraints.p[k]

function ConstraintSet(constraints, N)
	p = zeros(N)
	c_max = zeros(length(constraints))
	for con in constraints
		for k = 1:N
			p[k] += length(con, k)
		end
	end
	ConstraintSet(constraints, p, c_max)
end

function max_violation!(conSet::ConstraintSet{T}) where T
	for i in eachindex(conSet.constraints)
		max_violation!(conSet.constraints[i])
		c_max = conSet.c_max::Vector{T}
		conSet.c_max[i] = maximum(c_max)
	end
end




struct CircleConstraint{T,P} <: AbstractConstraint{Inequality}
	n::Int
	m::Int
	x::SVector{P,T}
	y::SVector{P,T}
	radius::T
	CircleConstraint(n::Int, m::Int, xc::SVector{P,T}, yc::SVector{P,T}, radius::T) where {T,P} =
		 new{T,P}(n,m,xc,yc,radius)
end

Base.size(con::CircleConstraint{T,P}) where {T,P} = (con.n, con.m, P)

function evaluate(con::CircleConstraint{T,P}, x, u) where {T,P}
	xc = con.x
	yc = con.y
	r = con.radius
	-(x[1] - xc).^2 - (x[2] - yc).^2 + r^2
end


struct NormConstraint{T} <: AbstractConstraint{Equality}
	n::Int
	m::Int
	p::Int
	val::T
end

Base.size(con::NormConstraint) = (con.n, con.m, con.p)

function evaluate(con::NormConstraint, x, u)
	return @SVector [norm(x) - con.val]
end


struct BoundConstraint{T,P,PN,NM,PNM} <: AbstractConstraint{Inequality}
	n::Int
	m::Int
	z_max::SVector{NM,T}
	z_min::SVector{NM,T}
	b::SVector{P,T}
	B::SMatrix{P,NM,T,PNM}
	active_N::SVector{PN,Int}
end

function BoundConstraint(n, m; x_max=zeros(n)*Inf, x_min=zeros(n)*-Inf,
		u_max=zeros(m)*Inf, u_min=zeros(m)*-Inf)
	z_max = [x_max; u_max]
	z_min = [x_min; u_min]
	b = [-z_max; z_min]
	bN = [x_max; u_max*Inf; x_min; -u_min*Inf]

	active = isfinite.(b)
	active_N = isfinite.(bN)
	p = sum(active)
	pN = sum(active_N)

	inds = SVector{p}(findall(active))
	inds_N = SVector{pN}(findall(active_N[active]))

	B = SMatrix{2(n+m), n+m}([1.0I(n+m); -1.0I(n+m)])


	BoundConstraint(n, m, z_max, z_min, b[inds], B[inds,:], inds_N)
end

Base.size(bnd::BoundConstraint{T,P,PN,NM,PNM}) where {T,P,PN,NM,PNM} = (bnd.n, bnd.m, P)

function evaluate(bnd::BoundConstraint{T,P,PN,NM,PNM}, x, u) where {T,P,PN,NM,PNM}
	bnd.B*SVector{NM}([x; u]) + bnd.b
end

function evaluate(bnd::BoundConstraint{T,P,PN,NM,PNM}, x::SVector{n,T}) where {T,P,PN,NM,PNM,n}
	ix = SVector{n}(1:n)
	B_N = bnd.B[bnd.active_N, ix]
	b_N = bnd.b[bnd.active_N]
	B_N*x + b_N
end

function jacobian(bnd::BoundConstraint, x, u)
	bnd.B
end

function jacobian(bnd::BoundConstraint, x::SVector{n,T}) where{n,T}
	ix = SVector{n}(1:n)
	bnd.B[bnd.active_N, ix]
end

function generate_jacobian(con::C) where {C<:AbstractConstraint}
	n,m = size(con)
	ix = SVector{n}(1:n)
	iu = SVector{m}(n .+ (1:m))
    f_aug(z) = evaluate(con, z[ix], z[iu])
    # ix,iu = 1:n, n .+ (1:m)
    # f_aug(z) = evaluate(con, view(z,ix), view(z,iu))
    ∇f(z) = ForwardDiff.jacobian(f_aug,z)
    ∇f(x::SVector,u::SVector) = ∇f([x;u])
    ∇f(x,u) = begin
        z = zeros(n+m)
        z[ix] = x
        z[iu] = u
        ∇f(z)
    end
    @eval begin
        jacobian(con::$(C), x, u) = $(∇f)(x, u)
        jacobian(con::$(C), z) = $(∇f)(z)
		jacobian(con::$(C), z::KnotPoint) = $(∇f)(z.z)
    end
end
