export
	ImplicitDynamics,
	ExplicitDynamics,
	GoalConstraint,
	StaticBoundConstraint,
	CircleConstraint,
	NormConstraint

abstract type ConstraintSense end
abstract type Equality <: ConstraintSense end
abstract type Inequality <: ConstraintSense end
abstract type Null <: ConstraintSense end

abstract type GeneralConstraint end
abstract type AbstractConstraint{S<:ConstraintSense} <: GeneralConstraint end

abstract type ConstraintType end
abstract type Stage <: ConstraintType end
abstract type State <: Stage end
abstract type Control <: Stage end
abstract type Coupled <: ConstraintType end
abstract type Dynamical <: Coupled end
abstract type CoupledState <: Coupled end
abstract type CoupledControl <: Coupled end
abstract type General <: ConstraintType end
abstract type GeneralState <: General end
abstract type GeneralControl <: General end

abstract type AbstractStaticConstraint{S<:ConstraintSense,W<:ConstraintType,P} <: GeneralConstraint end


"Returns the width of band imposed by the constraint"
width(con::AbstractStaticConstraint{S,Stage}) where S = sum(size(con)[1:2])
width(con::AbstractStaticConstraint{S,State}) where S = size(con)[1]
width(con::AbstractStaticConstraint{S,Control}) where S = size(con)[2]
width(con::AbstractStaticConstraint{S,Coupled}) where S = 2*sum(size(con)[1:2])
width(con::AbstractStaticConstraint{S,Dynamical}) where S = begin n,m = size(con); 2n + m end
width(con::AbstractStaticConstraint{S,CoupledState}) where S = 2*size(con)[1]
width(con::AbstractStaticConstraint{S,CoupledControl}) where S = 2*size(con)[2]
width(con::AbstractStaticConstraint{S,<:General}) where S = Inf

@inline length(con::AbstractStaticConstraint) = size(con)[3]
upper_bound(con::AbstractStaticConstraint{Inequality,W,P}) where {P,W} = @SVector zeros(P)
lower_bound(con::AbstractStaticConstraint{Inequality,W,P}) where {P,W} = -Inf*@SVector ones(P)
upper_bound(con::AbstractStaticConstraint{Equality,W,P}) where {P,W} = @SVector zeros(P)
lower_bound(con::AbstractStaticConstraint{Equality,W,P}) where {P,W} = @SVector zeros(P)

@inline is_bound(con::AbstractStaticConstraint) = false

@inline precompute(con::AbstractStaticConstraint, Z::Traj) = nothing

"""
Default evaluation of a constraint over and entire trajectory.
This should be the method used to evaluate constraints.
Some constraints may choose to replace this generic method (e.g. dynamics constraints)
"""
function evaluate!(vals::Vector{<:AbstractVector}, con::AbstractStaticConstraint{P,<:Stage},
		Z::Traj, inds=1:length(Z)) where P
	for (i,k) in enumerate(inds)
		vals[i] = evaluate(con, Z[k])
	end
end

function evaluate!(vals::Vector{<:AbstractVector}, con::AbstractStaticConstraint{P,<:Coupled},
		Z::Traj, inds=1:length(Z)-1) where P
	for (i,k) in enumerate(inds)
		vals[i] = evaluate(con, Z[k+1], Z[k])
	end
end

function jacobian!(∇c::Vector{<:AbstractMatrix}, con::AbstractStaticConstraint{P,<:Stage},
		Z::Traj, inds=1:length(Z)) where P
	for (i,k) in enumerate(inds)
		∇c[i] = jacobian(con, Z[k])
	end
end

function jacobian!(∇c::Vector{<:AbstractMatrix}, con::AbstractStaticConstraint{P,<:Coupled},
	Z::Traj, inds=1:length(Z)-1) where P
	for (i,k) in enumerate(inds)
		∇c[i] = jacobian(con, Z[k+1], Z[k])
	end
end

for method in [:evaluate, :jacobian]
	@eval begin
			@inline $(method)(con::AbstractStaticConstraint{P,Stage},   Z::KnotPoint) where P = $(method)(con, state(Z), control(Z))
			@inline $(method)(con::AbstractStaticConstraint{P,State},   Z::KnotPoint) where P = $(method)(con, state(Z))
			@inline $(method)(con::AbstractStaticConstraint{P,Control}, Z::KnotPoint) where P = $(method)(con, control(Z))

			@inline $(method)(con::AbstractStaticConstraint{P,Coupled}, Z′::KnotPoint, Z::KnotPoint) where P =
				$(method)(con, state(Z′), control(Z′), state(Z), control(Z))
			@inline $(method)(con::AbstractStaticConstraint{P,Dynamical}, Z′::KnotPoint, Z::KnotPoint) where P =
				$(method)(con, state(Z′), state(Z), control(Z))
	end
end




############################################################################################
#                              DYNAMICS CONSTRAINTS										   #
############################################################################################


abstract type AbstractDynamicsConstraint{W<:Coupled,P} <: AbstractStaticConstraint{Equality,W,P} end
Base.size(con::AbstractDynamicsConstraint) = begin n,m = size(con.model); return n,m,n end


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

# Hermite Simpson
function evaluate!(vals::Vector{<:AbstractVector}, con::DynamicsConstraint{HermiteSimpson},
		Z::Traj, inds=1:length(Z)-1)
	N = length(Z)
	model = con.model
	fVal = con.fVal
	xMid = con.xMid

	for k = inds.start:inds.stop+1
		fVal[k] = dynamics(model, Z[k])
	end
	for k in inds
		xMid[k] = (state(Z[k]) + state(Z[k+1]))/2 + Z[k].dt/8 * (fVal[k] - fVal[k+1])
	end
	for k in inds
		Um = (control(Z[k]) + control(Z[k+1]))*0.5
		fValm = dynamics(model, xMid[k], Um)
		vals[k] = state(Z[k]) - state(Z[k+1]) + Z[k].dt*(fVal[k] + 4*fValm + fVal[k+1])/6
	end
end

function jacobian!(∇c::Vector{<:AbstractMatrix}, con::DynamicsConstraint{HermiteSimpson,L,T,n},
		Z::Traj, inds=1:length(Z)-1) where {L,T,n}
	N = length(Z)
	model = con.model
	∇f = con.∇f
	xMid = con.xMid
	In = Diagonal(@SVector ones(n))

	xi = Z[1]._x
	ui = Z[1]._u

	# Compute dynamics Jacobian at each knot point
	for k = inds.start:inds.stop+1
		∇f[k] = jacobian(model, Z[k].z)
	end

	for k in inds
		Um = (control(Z[k]) + control(Z[k+1]))*0.5
		Fm = jacobian(model, [xMid[k]; Um])
		A1 = ∇f[k][xi,xi]
		B1 = ∇f[k][xi,ui]
		Am = Fm[xi,xi]
		Bm = Fm[xi,ui]
		A2 = ∇f[k+1][xi,xi]
		B2 = ∇f[k+1][xi,ui]
		dt = Z[k].dt
		A = dt/6*(A1 + 4Am*( dt/8*A1 + In/2)) + In
		B = dt/6*(B1 + 4Am*( dt/8*B1) + 2Bm)
		C = dt/6*(A2 + 4Am*(-dt/8*A2 + In/2)) - In
		D = dt/6*(B2 + 4Am*(-dt/8*B2) + 2Bm)
		∇c[k] = [A B C D]
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

############################################################################################
#                              CUSTOM CONSTRAINTS 										   #
############################################################################################

struct GoalConstraint{T,P,N,L} <: AbstractStaticConstraint{Equality,State,P}
	n::Int
	m::Int
	xf::SVector{P,T}
	Ix::SMatrix{P,N,T,L}
	inds::SVector{P,Int}
end

function GoalConstraint(n::Int, m::Int, xf::AbstractVector, inds=SVector{n}(1:n))
	Ix = Diagonal(@SVector ones(n))
	Ix = SMatrix{n,n}(Matrix(1.0I,n,n))
	Ix = Ix[inds,:]
	p = length(inds)
	GoalConstraint(n, m, SVector{p}(xf[inds]), Ix, inds)
end

size(con::GoalConstraint{T,N}) where {T,N} = (con.n, con.m,N)
evaluate(con::GoalConstraint, x::SVector) = x[con.inds] - con.xf
jacobian(con::GoalConstraint, z::KnotPoint) = con.Ix

struct LinearConstraint{S,W<:Stage,T,P,N,L} <: AbstractStaticConstraint{S,W,P}
	n::Int
	m::Int
	A::SMatrix{P,N,T,L}
	b::SMatrix{P,T}
end

size(con::LinearConstraint{S,W,T,P}) where {S,W,T,P} = (con.n, con.m, P)
evaluate(con::LinearConstraint,x) = con.A*x - con.b
jacobian(con::LinearConstraint,x) = con.A


struct CircleConstraint{T,P} <: AbstractStaticConstraint{Inequality,Stage,P}
	n::Int
	m::Int
	x::SVector{P,T}
	y::SVector{P,T}
	radius::SVector{P,T}
	CircleConstraint(n::Int, m::Int, xc::SVector{P,T}, yc::SVector{P,T}, radius::SVector{P,T}) where {T,P} =
		 new{T,P}(n,m,xc,yc,radius)
end

Base.size(con::CircleConstraint{T,P}) where {T,P} = (con.n, con.m, P)

function evaluate(con::CircleConstraint{T,P}, x, u) where {T,P}
	xc = con.x
	yc = con.y
	r = con.radius
	-(x[1] .- xc).^2 - (x[2] .- yc).^2 + r.^2
end


struct SphereConstraint{T,P} <: AbstractStaticConstraint{Inequality,Stage,P}
	n::Int
	m::Int
	x::SVector{P,T}
	y::SVector{P,T}
	z::SVector{P,T}
	radius::SVector{P,T}
	SphereConstraint(n::Int, m::Int, xc::SVector{P,T}, yc::SVector{P,T}, zc::SVector{P,T},
			radius::SVector{P,T}) where {T,P} = new{T,P}(n,m,xc,yc,zc,radius)
end

Base.size(con::SphereConstraint{T,P}) where {T,P} = (con.n, con.m, P)

function evaluate(con::SphereConstraint{T,P}, x, u) where {T,P}
	xc = con.x
	yc = con.y
	zc = con.z
	r = con.radius

	-((x[1] - xc).^2 + (x[2] - yc).^2 + (x[3] - zc).^2 - r.^2)
	# -(x[1] - xc).^2 .- (x[2] - yc).^2 .- (x[3] - zc).^2 .+ r.^2
end

struct ControlNorm{S,T} <: AbstractStaticConstraint{S,Control,1}
	n::Int
	m::Int
	val::T
	function ControlNorm{S}(n::Int,m::Int,val::T) where {S,T}
		@assert val >= 0
		new{S,T}(n,m,val)
	end
end
Base.size(con::ControlNorm) = con.n, con.m

function evaluate(con::ControlNorm, u)
	return @SVector [u'u - con.val^2]
end



struct NormConstraint{S,T} <: AbstractStaticConstraint{S,Stage,1}
	n::Int
	m::Int
	val::T
end

Base.size(con::NormConstraint{T}) where {T} = (con.n, con.m, 1)

function evaluate(con::NormConstraint, x, u)
	return @SVector [norm(x) - con.val]
end


struct StaticBoundConstraint{T,P,PN,NM,PNM} <: AbstractStaticConstraint{Inequality,Stage,P}
	n::Int
	m::Int
	z_max::SVector{NM,T}
	z_min::SVector{NM,T}
	b::SVector{P,T}
	B::SMatrix{P,NM,T,PNM}
	active_N::SVector{PN,Int}
end

function StaticBoundConstraint(n, m; x_max=Inf*(@SVector ones(n)), x_min=-Inf*(@SVector ones(n)),
		u_max=Inf*(@SVector ones(m)), u_min=-Inf*(@SVector ones(m)))
	# Check and convert bounds
	x_max, x_min = checkBounds(Val(n), x_max, x_min)
	u_max, u_min = checkBounds(Val(m), u_max, u_min)

	# Concatenate bounds
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

	StaticBoundConstraint(n, m, z_max, z_min, b[inds], B[inds,:], inds_N)
end

function checkBounds(::Val{N}, u::AbstractVector, l::AbstractVector) where N
	if all(u .>= l)
		return SVector{N}(u), SVector{N}(l)
	else
		throw(ArgumentError("Upper bounds must be greater than or equal to lower bounds"))
	end
end

checkBounds(sze::Val{N}, u::Real, l::Real) where N =
	checkBounds(sze, (@SVector fill(u,N)), (@SVector fill(l,N)))
checkBounds(sze::Val{N}, u::AbstractVector, l::Real) where N =
	checkBounds(sze, u, (@SVector fill(l,N)))
checkBounds(sze::Val{N}, u::Real, l::AbstractVector) where N =
	checkBounds(sze, (@SVector fill(u,N)), l)


Base.size(bnd::StaticBoundConstraint{T,P,PN,NM,PNM}) where {T,P,PN,NM,PNM} = (bnd.n, bnd.m, P)
is_bound(::StaticBoundConstraint) = true
lower_bound(bnd::StaticBoundConstraint) = bnd.z_min
upper_bound(bnd::StaticBoundConstraint) = bnd.z_max


function evaluate(bnd::StaticBoundConstraint{T,P,PN,NM,PNM}, x, u) where {T,P,PN,NM,PNM}
	bnd.B*SVector{NM}([x; u]) + bnd.b
end

function evaluate(bnd::StaticBoundConstraint{T,P,PN,NM,PNM}, x::SVector{n,T}) where {T,P,PN,NM,PNM,n}
	ix = SVector{n}(1:n)
	B_N = bnd.B[bnd.active_N, ix]
	b_N = bnd.b[bnd.active_N]
	B_N*x + b_N
end

function jacobian(bnd::StaticBoundConstraint, z::KnotPoint)
	bnd.B
end

# function jacobian(bnd::StaticBoundConstraint, x::SVector{n,T}) where{n,T}
# 	ix = SVector{n}(1:n)
# 	bnd.B[bnd.active_N, ix]
# end



function generate_jacobian(con::C) where {C<:GeneralConstraint}
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
