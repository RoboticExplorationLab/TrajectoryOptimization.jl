export
	ImplicitDynamics,
	ExplicitDynamics,
	GoalConstraint,
	StaticBoundConstraint,
	CircleConstraint,
	NormConstraint

export
	ConstraintSense,
	Inequality,
	Equality,
	Stage,
	State,
	Control,
	Coupled,
	Dynamical

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

# Getters
contype(::AbstractStaticConstraint{S,W}) where {S,W} = W
sense(::AbstractStaticConstraint{S}) where S = S

"Returns the width of band imposed by the constraint"
width(con::AbstractStaticConstraint{S,Stage}) where S = state_dim(con) + control_dim(con)
width(con::AbstractStaticConstraint{S,State}) where S = state_dim(con)
width(con::AbstractStaticConstraint{S,Control}) where S = control_dim(con)
width(con::AbstractStaticConstraint{S,Coupled}) where S = 2*(state_dim(con) + control_dim(con))
width(con::AbstractStaticConstraint{S,Dynamical}) where S = 2*state_dim(con) + control_dim(con)
width(con::AbstractStaticConstraint{S,CoupledState}) where S = 2*state_dim(con)
width(con::AbstractStaticConstraint{S,CoupledControl}) where S = 2*control_dim(con)
width(con::AbstractStaticConstraint{S,<:General}) where S = Inf

upper_bound(con::AbstractStaticConstraint{Inequality,W,P}) where {P,W} = @SVector zeros(P)
lower_bound(con::AbstractStaticConstraint{Inequality,W,P}) where {P,W} = -Inf*@SVector ones(P)
upper_bound(con::AbstractStaticConstraint{Equality,W,P}) where {P,W} = @SVector zeros(P)
lower_bound(con::AbstractStaticConstraint{Equality,W,P}) where {P,W} = @SVector zeros(P)
@inline is_bound(con::AbstractStaticConstraint) = false

@inline check_dims(con::AbstractStaticConstraint{S,State},n,m) where S = state_dim(con) == n
@inline check_dims(con::AbstractStaticConstraint{S,Control},n,m) where S = control_dim(con) == m
@inline function check_dims(con::AbstractStaticConstraint{S,W},n,m) where {S,W<:ConstraintType}
	state_dim(con) == n && control_dim(con) == m
end

control_dims(::AbstractStaticConstraint{S,State}) where S =
	throw(ErrorException("Cannot get control dimension from a state-only constraint"))

state_dims(::AbstractStaticConstraint{S,Control}) where S =
	throw(ErrorException("Cannot get state dimension from a control-only constraint"))

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
			@inline $(method)(con::AbstractStaticConstraint{S,Stage},   Z::KnotPoint) where S = $(method)(con, state(Z), control(Z))
			@inline $(method)(con::AbstractStaticConstraint{S,State},   Z::KnotPoint) where S = $(method)(con, state(Z))
			@inline $(method)(con::AbstractStaticConstraint{S,Control}, Z::KnotPoint) where S = $(method)(con, control(Z))

			@inline $(method)(con::AbstractStaticConstraint{S,Coupled}, Z′::KnotPoint, Z::KnotPoint) where S =
				$(method)(con, state(Z′), control(Z′), state(Z), control(Z))
			@inline $(method)(con::AbstractStaticConstraint{S,Dynamical}, Z′::KnotPoint, Z::KnotPoint) where S =
				$(method)(con, state(Z′), state(Z), control(Z))
	end
end


function jacobian(con::AbstractStaticConstraint{P,W}, x::SVector{N}) where {P,N,W<:Union{State,Control}}
	eval_c(x) = evaluate(con, x)
	ForwardDiff.jacobian(eval_c, x)
end


############################################################################################
#                              DYNAMICS CONSTRAINTS										   #
############################################################################################


abstract type AbstractDynamicsConstraint{W<:Coupled,P} <: AbstractStaticConstraint{Equality,W,P} end
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

state_dim(::GoalConstraint{T,P,N}) where {T,P,N} = N
Base.length(::GoalConstraint{T,P}) where {T,P} = P
evaluate(con::GoalConstraint, x::SVector) = x[con.inds] - con.xf
jacobian(con::GoalConstraint, z::KnotPoint) = con.Ix

struct LinearConstraint{S,W<:Union{State,Control},P,N,L,T} <: AbstractStaticConstraint{S,W,P}
	n::Int
	m::Int
	A::SMatrix{P,N,T,L}
	b::SMatrix{P,T}
end

state_dim(::LinearConstraint{S,State,P,N}) where {S,P,N} = N
control_dim(::LinearConstraint{S,Control,P,N}) where {S,P,N} = N
Base.length(::LinearConstraint{S,W,P}) where {S,W,P} = P
evaluate(con::LinearConstraint,x) = con.A*x - con.b
jacobian(con::LinearConstraint,x) = con.A


struct CircleConstraint{T,P} <: AbstractStaticConstraint{Inequality,State,P}
	n::Int
	m::Int
	x::SVector{P,T}
	y::SVector{P,T}
	radius::SVector{P,T}
	CircleConstraint(n::Int, m::Int, xc::SVector{P,T}, yc::SVector{P,T}, radius::SVector{P,T}) where {T,P} =
		 new{T,P}(n,m,xc,yc,radius)
end
state_dim(con::CircleConstraint) = con.n
Base.length(::CircleConstraint{T,P}) where {T,P} = P

function evaluate(con::CircleConstraint{T,P}, x::SVector) where {T,P}
	xc = con.x
	yc = con.y
	r = con.radius
	-(x[1] .- xc).^2 - (x[2] .- yc).^2 + r.^2
end

function Plots.plot!(con::CircleConstraint{T,P}; color=:red, kwargs...) where {T,P}
	for i = 1:P
		x,y,r = con.x[i], con.y[i], con.radius[i]
		plot_circle!((x,y),r; kwargs..., label="", color=color, linecolor=color)
	end
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

state_dim(con::SphereConstraint) = con.n
Base.length(::SphereConstraint{T,P}) where {T,P} = P

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

function evaluate(con::ControlNorm, u)
	return @SVector [u'u - con.val^2]
end



struct NormConstraint{S,W<:Union{State,Control},T} <: AbstractStaticConstraint{S,W,1}
	dim::Int
	val::T
end
state_dim(con::NormConstraint{S,State}) where S = con.dim
control_dim(con::NormConstraint{S,Control}) where S = con.dim
Base.length(con::NormConstraint) = 1

function evaluate(con::NormConstraint, x)
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


state_dim(con::StaticBoundConstraint) = con.n
control_dim(con::StaticBoundConstraint) = con.m
Base.length(bnd::StaticBoundConstraint{T,P}) where {T,P} = P
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


struct InfeasibleConstraint{N,M} <: AbstractStaticConstraint{Equality, Control, M} end

InfeasibleConstraint(model::InfeasibleModel{N,M}) where {N,M} = InfeasibleConstraint{N,M}()
InfeasibleConstraint(n::Int, m::Int) = InfeasibleConstraint{n,m}()
control_dim(::InfeasibleConstraint{N,M}) where {N,M} = N+M
Base.length(::InfeasibleConstraint{N,M}) where {N,M} = N

@generated function evaluate(con::InfeasibleConstraint{N,M}, u::SVector) where {N,M}
    _u = SVector{M}(1:M)
    _ui = SVector{N}((1:N) .+ M)
	quote
        ui = u[$_ui] # infeasible controls
	end
end

@generated function jacobian(con::InfeasibleConstraint{N,M}, u::SVector) where {N,M}
	Iu = [(@SMatrix zeros(N,M)) Diagonal(@SVector ones(N))]
	return :($Iu)
end



struct IndexedConstraint{S,W,P,N,M,NM,Bx,Bu,C} <: AbstractStaticConstraint{S,W,P}
	n::Int  # new dimension
	m::Int  # new dimension
	con::C
	ix::SVector{N,Int}
	iu::SVector{M,Int}
	z::KnotPoint{Float64,N,M,NM}
end

state_dim(con::IndexedConstraint{S,Union{Stage,State}}) where S = con.n
control_dim(con::IndexedConstraint{S,Union{Stage,Control}}) where S = con.m
Base.length(::IndexedConstraint{S,W,P}) where {S,W,P} = P

function IndexedConstraint(n,m,con::AbstractStaticConstraint{S,W,P},
		ix::SVector{N}, iu::SVector{M}) where {S,W,P,N,M}
	x = @SVector rand(N)
	u = @SVector rand(M)
	z = KnotPoint(x,u,0.0)
	Bx = ix[1]
	Bu = iu[1]
	IndexedConstraint{S,W,P,N,M,N+M,Bx,Bu,typeof(con)}(n,m,con,ix,iu,z)
end

function IndexedConstraint(n,m,con::AbstractStaticConstraint{S,W}) where {S,W}
	if W <: Union{State,CoupledState}
		m0 = m
	else
		m0 = control_dim(con)
	end
	if W<: Union{Control,CoupledControl}
		n0 = n
	else
		n0 = state_dim(con)
	end
	ix = SVector{n0}(1:n0)
	iu = SVector{m0}(1:m0)
	IndexedConstraint(n,m,con, ix, iu)
end

function evaluate(con::IndexedConstraint{S,<:Stage}, z::KnotPoint) where {S}
	x0 = state(z)[con.ix]
	u0 = control(z)[con.iu]
	con.z.z = [x0; u0]
	evaluate(con.con, con.z)
end

@generated function jacobian(con::IndexedConstraint{S,Stage,P,N0,M0,NM0,Bx,Bu},
		z::KnotPoint{T,N,M,NM}) where {S,P,N0,M0,NM0,Bx,Bu,T,N,M,NM}
    l1 = Bx-1
    l2 = N-(Bx+N0-1)
    l3 = Bu-1
    l4 = NM-(N+Bu+M0-1)

	∇c1 = @SMatrix zeros(P,l1)
	∇c2 = @SMatrix zeros(P,l2)
	∇c3 = @SMatrix zeros(P,l3)
	∇c4 = @SMatrix zeros(P,l4)

	ix = SVector{N0}(1:N0)
	iu = SVector{M0}((1:M0) .+ N0)
	quote
		x0 = state(z)[con.ix]
		u0 = control(z)[con.iu]
		con.z.z = [x0; u0]
		∇c = jacobian(con.con, con.z)
		A = ∇c[$ix,$ix]
		B = ∇c[$ix,$iu]
		[$∇c1 A $∇c2 $∇c3 B $∇c4]
	end
end

@generated function jacobian(con::IndexedConstraint{S,State,P,N0,M0,NM0,Bx,Bu},
		z::KnotPoint{T,N,M,NM}) where {S,P,N0,M0,NM0,Bx,Bu,T,N,M,NM}
    l1 = Bx-1
    l2 = N-(Bx+N0-1)

	∇c1 = @SMatrix zeros(P,l1)
	∇c2 = @SMatrix zeros(P,l2)

	ix = SVector{N0}(1:N0)
	iu = SVector{M0}((1:M0) .+ N0)
	quote
		x0 = state(z)[con.ix]
		u0 = control(z)[con.iu]
		con.z.z = [x0; u0]
		∇c = jacobian(con.con, con.z)
		[$∇c1 ∇c $∇c2]
	end
end

@generated function jacobian(con::IndexedConstraint{S,Control,P,N0,M0,NM0,Bx,Bu},
		z::KnotPoint{T,N,M,NM}) where {S,P,N0,M0,NM0,Bx,Bu,T,N,M,NM}
    l3 = Bu-1
    l4 = NM-(N+Bu+M0-1)

	∇c3 = @SMatrix zeros(P,l3)
	∇c4 = @SMatrix zeros(P,l4)

	quote
		x0 = state(z)[con.ix]
		u0 = control(z)[con.iu]
		con.z.z = [x0; u0]
		∇c = jacobian(con.con, con.z)
		[$∇c3 ∇c $∇c4]
	end
end


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
