using StaticArrays, ForwardDiff, BenchmarkTools, LinearAlgebra

abstract type ConstraintType end
abstract type Equality <: ConstraintType end
abstract type Inequality <: ConstraintType end
abstract type Null <: ConstraintType end
abstract type AbstractConstraint{S<:ConstraintType} end

n,m = 13,4
dt = 0.1
xs,us = (@SVector rand(n)), (@SVector rand(m))
x,u = Array(xs), Array(us)
zs = [xs;us]
z = [x;u]
Z = KnotPoint(x,u,dt)


struct CircleConstraint{T} <: AbstractConstraint{Inequality}
	n::Int
	m::Int
	p::Int
	r0::SVector{3,T}
	radius::T
end

Base.size(con::CircleConstraint) = (con.n, con.m, con.p)

function evaluate(con::CircleConstraint, x, u)
	p = con.r0
	r = con.radius
	return @SVector [-((x[1] - p[1])^2 + (x[2]-p[2])^2  - r^2)]
end

function evaluate(con::CircleConstraint, z)
	p = con.r0
	r = con.radius
	return @SVector [-((z[1] - p[1])^2 + (z[2]-p[2])^2  - r^2)]
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

function generate_jacobian(con::Function,n,m)
	ix = SVector{n}(1:n)
	iu = SVector{m}(n .+ (1:m))
    f_aug(z) = con(z[ix], z[iu])
    ∇f(z) = ForwardDiff.jacobian(f_aug,z)
    ∇f(x::SVector,u::SVector) = ∇f([x;u])
    ∇f(x,u) = begin
        z = zeros(n+m)
        z[ix] = x
        z[iu] = u
        ∇f(z)
    end
	∇con = Symbol("∇" * string(con))
    @eval begin
        $(∇con)(x, u) = $(∇f)(x, u)
        $(∇con)(z) = $(∇f)(z)
		$(∇con)(z::KnotPoint) = $(∇f)(z.z)
    end
	return ∇con

end

r0 = @SVector [0.5, 0.5, 0]
radius = 0.1
con = CircleConstraint(n, m, 1, r0, radius)
evaluate(con, x, u) == evaluate(con, Z.z)
@btime evaluate($con, $xs, $us)
@btime evaluate($con, $(Z.z))


generate_jacobian(con)
jacobian(con, zs)
jacobian(con, xs, us)
jacobian(con, Z)
@btime jacobian($con, $zs)
@btime jacobian($con, $xs, $us)
@btime jacobian($con, $Z)

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

	println("hi")

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

n,m = 3,2
dt = 0.1
xs,us = (@SVector rand(n)), (@SVector rand(m))
x,u = Array(xs), Array(us)
zs = [xs;us]
z = [x;u]

x_max = @SVector [3,2,Inf]
x_min = @SVector [-3,-2,-Inf]
u_max = @SVector [Inf, 0.5]
u_min = @SVector [0.0, -0.5]

bnd = BoundConstraint(n,m, x_max=x_max, x_min=x_min, u_max=u_max, u_min=u_min)

@btime evaluate($bnd, $xs, $us)
@btime evaluate($bnd, $xs)



# Custom constraints
n,m = 3,2
p1 = 3

struct GeneralConstraint{C} <: AbstractConstraint{C}
	n::Int
	m::Int
	p::Int
	c::Function
	∇c::Function
end

function GeneralConstraint{C}(n::Int, m::Int, p::Int, c::Function) where C
	∇c = eval(Symbol("∇" * string(c)))
	generate_jacobian(c, n, m)
	GeneralConstraint{C}(n,m,p,c,∇c)
end

evaluate(con::GeneralConstraint, x, u) = con.c(x,u)
evaluate(con::GeneralConstraint, z) = con.c(z)
jacobian(con::GeneralConstraint, x, u) = con.∇c(x, u)
jacobian(con::GeneralConstraint, z) = con.∇c(z)


function mycon(x, u)
	@SVector [x[1]^2 + x[2]^2 - 5, u[1] - 1, u[2] - 1]
end
generate_jacobian(mycon, n, m)

eq1 = GeneralConstraint{Equality}(n,m,3,mycon, ∇mycon)

function mycon2(x, u)
	@SVector [x[1]^2 + x[2]^2 - 5, u[1] - 1, u[2] - 1]
end
eq2 = GeneralConstraint{Equality}(n,m,3,mycon2)

@btime evaluate($eq2,$xs,$us)
@btime jacobian($eq2,$zs)


@btime evaluate($eq1, $xs, $us)
jacobian(eq1, xs, us)
@btime jacobian($eq1, $zs)
