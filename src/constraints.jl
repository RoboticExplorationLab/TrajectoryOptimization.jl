# export
# 	GoalConstraint,
# 	BoundConstraint,
# 	CircleConstraint,
# 	SphereConstraint,
# 	NormConstraint,
# 	LinearConstraint,
# 	VariableBoundConstraint,
# 	QuatNormConstraint,
# 	QuatSlackConstraint

import RobotDynamics: state_dim, control_dim


############################################################################################
#                              GOAL CONSTRAINTS 										   #
############################################################################################

"""
	GoalConstraint{P,T}

Constraint of the form ``x_g = a``, where ``x_g`` can be only part of the state
vector.

# Constructors:
```julia
GoalConstraint(xf::AbstractVector)
GoalConstraint(xf::AbstractVector, inds)
```
where `xf` is an n-dimensional goal state. If `inds` is provided,
only `xf[inds]` will be used.
"""
struct GoalConstraint{P,T} <: StateConstraint
	n::Int
	xf::SVector{P,T}
	inds::SVector{P,Int}
end

function GoalConstraint(xf::AbstractVector, inds=1:length(xf))
	n = length(xf)
	p = length(inds)
	xf = SVector{n}(xf)
	inds = SVector{p}(inds)
	GoalConstraint(xf, inds)
end

function GoalConstraint(xf::SVector{n}, inds::SVector{p,Int}) where {n,p}
	GoalConstraint(length(xf), SVector{p}(xf[inds]), inds)
end

@inline sense(::GoalConstraint) = Equality()
@inline Base.length(con::GoalConstraint{P}) where P = P
@inline state_dim(con::GoalConstraint) = con.n
evaluate(con::GoalConstraint, x::SVector) = x[con.inds] - con.xf
function jacobian!(∇c, con::GoalConstraint, z::KnotPoint)
	T = eltype(∇c)
	for (i,j) in enumerate(con.inds)
		∇c[i,j] = one(T)
	end
	return true
end

function change_dimension(con::GoalConstraint, n::Int, m::Int, xi=1:n, ui=1:m)
	GoalConstraint(n, con.xf, xi[con.inds])
end


############################################################################################
#                              LINEAR CONSTRAINTS 										   #
############################################################################################
"""
	LinearConstraint{S,P,W,T}

Linear constraint of the form ``Ay - b \\{\\leq,=\\} 0`` where ``y`` may be either the
state or controls (but not a combination of both).

# Constructor: ```julia
LinearConstraint{S,W}(n,m,A,b)
```
where `W <: Union{State,Control}`.
"""
struct LinearConstraint{S,P,W,T} <: StageConstraint
	n::Int
	m::Int
	A::SizedMatrix{P,W,T,2}
	b::SVector{P,T}
	sense::S
	inds::SVector{W,Int}
	function LinearConstraint(n::Int, m::Int, A::StaticMatrix{P,W,T}, b::StaticVector{P,T},
			sense::ConstraintSense, inds=1:n+m) where {P,W,T}
		@assert length(inds) == W
		inds = SVector{W}(inds)
		new{typeof(sense),P,W,T}(n,m,A,b,sense,inds)
	end
end

function LinearConstraint(n::Int, m::Int, A::AbstractMatrix, b::AbstractVector,
		sense::S, inds=1:n+m) where {S<:ConstraintSense}
	@assert size(A,1) == length(b)
	p,q = size(A)
	A = SizedMatrix{p,q}(A)
	b = SVector{p}(b)
	LinearConstraint(n,m, A, b, sense, inds)
end


@inline sense(con::LinearConstraint) = con.sense
@inline Base.length(con::LinearConstraint{<:Any,P}) where P = P
@inline state_dim(con::LinearConstraint) = con.n
@inline control_dim(con::LinearConstraint) = con.m
evaluate(con::LinearConstraint, z::AbstractKnotPoint) = con.A*z.z[con.inds] .- con.b
function jacobian!(∇c, con::LinearConstraint, z::AbstractKnotPoint)
	∇c[:,con.inds] .= con.A
	return true
end

function change_dimension(con::LinearConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
	inds0 = [ix; n .+ iu]  # indices of original z in new z
	inds = inds0[con.inds] # indices of elements in new z
	LinearConstraint(n, m, con.A, con.b, con.sense, inds)
end

############################################################################################
#                              CIRCLE/SPHERE CONSTRAINTS 								   #
############################################################################################
"""
	CircleConstraint{P,T}

Constraint of the form
`` (x - x_c)^2 + (y - y_c)^2 \\leq r^2 ``
where ``x``, ``y`` are given by `x[xi]`,`x[yi]`, ``(x_c,y_c)`` is the center
of the circle, and ``r`` is the radius.

# Constructor:
```julia
CircleConstraint(n, xc::SVector{P}, yc::SVector{P}, radius::SVector{P}, xi=1, yi=2)
```
"""
struct CircleConstraint{P,T} <: StateConstraint
	n::Int
	x::SVector{P,T}
	y::SVector{P,T}
	radius::SVector{P,T}
	xi::Int  # index of x-state
	yi::Int  # index of y-state
	CircleConstraint(n::Int, xc::SVector{P,T}, yc::SVector{P,T}, radius::SVector{P,T},
			xi=1, yi=2) where {T,P} =
		 new{P,T}(n,xc,yc,radius,xi,yi)
end
state_dim(con::CircleConstraint) = con.n

function evaluate(con::CircleConstraint, X::StaticVector)
	xc = con.x
	yc = con.y
	r = con.radius
	x = X[con.xi]
	y = X[con.yi]
	-(x .- xc).^2 - (y .- yc).^2 + r.^2
end

function jacobian!(∇c, con::CircleConstraint{P}, X::SVector) where P
	xc = con.x; xi = con.xi
	yc = con.y; yi = con.yi
	x = X[xi]
	y = X[yi]
	r = con.radius
	∇f(x,xc) = -2*(x - xc)
	for i = 1:P
		∇c[i,xi] = ∇f(x, xc[i])
		∇c[i,yi] = ∇f(y, yc[i])
	end
	return false
end

@inline Base.length(::CircleConstraint{P}) where P = P
@inline sense(::CircleConstraint) = Inequality()

function change_dimension(con::CircleConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
	CircleConstraint(n, con.x, con.y, con.radius, ix[con.xi], ix[con.yi])
end

"""
	SphereConstraint{P,T}

Constraint of the form
`` (x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 \\leq r^2 ``
where ``x``, ``y``, ``z`` are given by `x[xi]`,`x[yi]`,`x[zi]`, ``(x_c,y_c,z_c)`` is the center
of the sphere, and ``r`` is the radius.

# Constructor:
```
SphereConstraint(n, xc::SVector{P}, yc::SVector{P}, zc::SVector{P},
	radius::SVector{P}, xi=1, yi=2, zi=3)
```
"""
struct SphereConstraint{P,T} <: StateConstraint
	n::Int
	x::SVector{P,T}
	y::SVector{P,T}
	z::SVector{P,T}
	xi::Int
	yi::Int
	zi::Int
	radius::SVector{P,T}
	SphereConstraint(n::Int, xc::SVector{P,T}, yc::SVector{P,T}, zc::SVector{P,T},
			radius::SVector{P,T}, xi=1, yi=2, zi=3) where {T,P} =
			new{P,T}(n,xc,yc,zc,xi,yi,zi,radius)
end

@inline state_dim(con::SphereConstraint) = con.n
@inline sense(::SphereConstraint) = Inequality()
@inline Base.length(::SphereConstraint{P}) where P = P

function evaluate(con::SphereConstraint, x::SVector)
	xc = con.x; xi = con.xi
	yc = con.y; yi = con.yi
	zc = con.z; zi = con.zi
	r = con.radius

	-((x[xi] .- xc).^2 + (x[yi] .- yc).^2 + (x[zi] .- zc).^2 - r.^2)
end

function jacobian!(con::SphereConstraint, X::SVector)
	xc = con.x; xi = con.xi
	yc = con.y; yi = con.yi
	zc = con.z; zi = con.zi
	x = X[xi]
	y = X[yi]
	z = X[zi]
	r = con.radius
	∇f(x,xc) = -2*(x - xc)
	for i = 1:P
		∇c[i,xi] = ∇f(x, xc[i])
		∇c[i,yi] = ∇f(y, yc[i])
		∇c[i,zi] = ∇f(z, zc[i])
	end
	return false
end

function change_dimension(con::SphereConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
	SphereConstraint(n, con.x, con.y, con.z, con.radius, ix[con.xi], ix[con.yi], ix[con.zi])
end

############################################################################################
#  								SELF-COLLISION CONSTRAINT 								   #
############################################################################################

struct CollisionConstraint{D} <: StateConstraint
	n::Int
    x1::SVector{D,Int}
    x2::SVector{D,Int}
    radius::Float64
end

@inline state_dim(con::CollisionConstraint) = con.n
@inline sense(::CollisionConstraint) = Inequality()
@inline Base.length(::CollisionConstraint) = 1

function evaluate(con::CollisionConstraint, x::SVector)
    x1 = x[con.x1]
    x2 = x[con.x2]
    d = x1 - x2
    @SVector [con.radius^2 - d'd]
end

function jacobian!(∇c, con::CollisionConstraint, x::SVector)
    x1 = x[con.x1]
    x2 = x[con.x2]
    d = x1 - x2
	∇x1 = -2d
	∇x2 =  2d
	∇c[1,con.x1] .= ∇x1
	∇c[1,con.x2] .= ∇x2
	return false
end

function change_dimension(con::CollisionConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
	CollisionConstraint(n, ix[con.x1], ix[con.x2], con.radius)
end

############################################################################################
#								NORM CONSTRAINT											   #
############################################################################################

"""
	NormConstraint{S,D,T}

Constraint of the form
``\\|y\\|^2 \\{\\leq,=\\} a``
where ``y`` is either a state or a control vector (but not both)

# Constructors:
```
NormConstraint{S,State}(n,a)
NormConstraint{S,Control}(m,a)
```
where `a` is the constant on the right-hand side of the equation.

# Examples:
```julia
NormConstraint{Equality,Control}(2,4.0)
```
creates a constraint equivalent to
``\\|u\\|^2 = 4.0`` for a problem with 2 controls.

```julia
NormConstraint{Inequality,State}(3, 2.3)
```
creates a constraint equivalent to
``\\|x\\|^2 \\leq 2.3`` for a problem with 3 states.
"""
struct NormConstraint{S,D,T} <: StageConstraint
	n::Int
	m::Int
	val::T
	sense::S
	inds::SVector{D,Int}
	function NormConstraint(n::Int, m::Int, val::T, sense::ConstraintSense,
			inds=SVector{n+m}(1:n+m)) where T
		if inds == :state
			inds = SVector{n}(1:n)
		elseif inds == :control
			inds = SVector{m}(n .+ (1:m))
		end
		@assert val ≥ 0 "Value must be greater than or equal to zero"
		new{typeof(sense),length(inds),T}(n,m,val,sense,inds)
	end
end

@inline state_dim(con::NormConstraint) = con.n
@inline control_dim(con::NormConstraint) = con.m
@inline sense(con::NormConstraint) = con.sense
@inline Base.length(::NormConstraint) = 1

function evaluate(con::NormConstraint, z::AbstractKnotPoint)
	x = z.z[con.inds]
	return @SVector [x'x - con.val]
end

function jacobian!(∇c, con::NormConstraint, z::AbstractKnotPoint)
	x = z.z[con.inds]
	∇c[1,con.inds] .= 2*x
	return false
end

function change_dimension(con::NormConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
	NormConstraint(n, m, con.val, con.sense, ix[con.inds])
end


############################################################################################
# 								COPY CONSTRAINT 										   #
############################################################################################

# struct CopyConstraint{K,W,S,P,N,M} <: AbstractConstraint{W,S,P}
# 	con::AbstractConstraint{W,S,P}
#     xinds::Vector{SVector{N,Int}}
#     uinds::Vector{SVector{M,Int}}
# end
#
# function evaluate(con::CopyConstraint{K}, z::KnotPoint)
# 	c = evaluate(con,)
# 	for 2 = 1:K
# 	end
# end


############################################################################################
# 								BOUND CONSTRAINTS 										   #
############################################################################################
"""
	BoundConstraint{P,NM,T}

Linear bound constraint on states and controls
# Constructors
```julia
BoundConstraint(n, m; x_min, x_max, u_min, u_max)
```
Any of the bounds can be ±∞. The bound can also be specifed as a single scalar, which applies the bound to all state/controls.
"""
struct BoundConstraint{P,NM,T} <: StageConstraint
	n::Int
	m::Int
	z_max::SVector{NM,T}
	z_min::SVector{NM,T}
	i_max::Vector{Int}
	i_min::Vector{Int}
	inds::SVector{P,Int}
end

function BoundConstraint(n, m; x_max=Inf*(@SVector ones(n)), x_min=-Inf*(@SVector ones(n)),
		u_max=Inf*(@SVector ones(m)), u_min=-Inf*(@SVector ones(m)))
	nm = n+m

	# Check and convert bounds
	x_max, x_min = checkBounds(n, x_max, x_min)
	u_max, u_min = checkBounds(m, u_max, u_min)

	# Concatenate bounds
	z_max = [x_max; u_max]
	z_min = [x_min; u_min]
	b = [-z_max; z_min]
	inds = findall(isfinite, b)
	inds = SVector{length(inds)}(inds)

	# Get linear indices of 1s of Jacobian
	a_max = findall(isfinite, z_max)
	a_min = findall(isfinite, z_min)
	u = length(a_max)
	l = length(a_min)
	carts_u = [CartesianIndex(i,   j) for (i,j) in enumerate(a_max)]
	carts_l = [CartesianIndex(i+u, j) for (i,j) in enumerate(a_min)]
	∇c = zeros(u+l, n+m)
	linds_u = LinearIndices(zeros(u+l,n+m))[carts_u]
	linds_l = LinearIndices(zeros(u+l,n+m))[carts_l]

	BoundConstraint(n, m, z_max, z_min, linds_u, linds_l, inds)
end

function con_label(con::BoundConstraint, ind::Int)
	i = con.inds[ind]
	n,m = state_dim(con), control_dim(con)
	if 1 <= i <= n
		return "x max $i"
	elseif n < i <= n + m
		j = i - n
		return "u max $j"
	elseif n + m < i <= 2n+m
		j = i - (n+m)
		return "x min $j"
	elseif 2n+m < i <= 2n+2m
		j = i - (2n+m)
		return "u min $j"
	else
		throw(BoundsError())
	end
end

function checkBounds(n::Int, u::AbstractVector, l::AbstractVector)
	if all(u .>= l)
		return SVector{n}(u), SVector{n}(l)
	else
		throw(ArgumentError("Upper bounds must be greater than or equal to lower bounds"))
	end
end

checkBounds(n::Int, u::Real, l::Real) =
	checkBounds(n, (@SVector fill(u,n)), (@SVector fill(l,n)))
checkBounds(n::Int, u::AbstractVector, l::Real) = checkBounds(n, u, (@SVector fill(l,N)))
checkBounds(n::Int, u::Real, l::AbstractVector) = checkBounds(n, (@SVector fill(u,N)), l)


@inline state_dim(con::BoundConstraint) = con.n
@inline control_dim(con::BoundConstraint) = con.m
@inline is_bound(::BoundConstraint) = true
@inline lower_bound(bnd::BoundConstraint) = bnd.z_min
@inline upper_bound(bnd::BoundConstraint) = bnd.z_max
@inline sense(::BoundConstraint) = Inequality()
@inline Base.length(con::BoundConstraint) = length(con.i_max) + length(con.i_min)


function evaluate(bnd::BoundConstraint, z::AbstractKnotPoint)
	[(z.z - bnd.z_max); (bnd.z_min - z.z)][bnd.inds]
end

function jacobian!(∇c, bnd::BoundConstraint{U,L}, z::AbstractKnotPoint) where {U,L}
	for i in bnd.i_max
		∇c[i]  = 1
	end
	for i in bnd.i_min
		∇c[i] = -1
	end
	return true
end

function change_dimension(con::BoundConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
	n0,m0 = con.n, con.m
	x_max = fill(Inf,n)
	x_min = fill(Inf,n)
	u_max = fill(Inf,m)
	u_min = fill(Inf,m)
	x_max[ix] = con.z_max[1:n0]
	x_min[ix] = con.z_min[1:n0]
	u_max[iu] = con.z_max[n0 .+ (1:m0)]
	u_min[iu] = con.z_min[n0 .+ (1:m0)]
	BoundConstraint(n, m, x_max=x_max, x_min=x_min, u_max=u_max, u_min=u_min)
end

############################################################################################
#  							VARIABLE BOUND CONSTRAINT 									   #
############################################################################################

# struct VariableBoundConstraint{T,P,NM,PNM} <: AbstractConstraint{Inequality,Stage,P}
# 	n::Int
# 	m::Int
# 	z_max::Vector{SVector{NM,T}}
# 	z_min::Vector{SVector{NM,T}}
# 	b::Vector{SVector{P,T}}
# 	B::SMatrix{P,NM,T,PNM}
# 	function VariableBoundConstraint(n::Int,m::Int,
# 			z_max::Vector{<:SVector{NM,T}}, z_min::Vector{<:SVector{NM,T}},
# 			b::Vector{<:SVector{P}}, B::SMatrix{P,NM,T,PNM}) where {T,P,PN,NM,PNM}
# 		new{T,P,NM,PNM}(n,m,z_max,z_min,b,B)
# 	end
# end
#
# state_dim(con::VariableBoundConstraint) = con.n
# control_dim(con::VariableBoundConstraint) = con.m
# is_bound(::VariableBoundConstraint) = true
#
# function evaluate!(vals::Vector{<:AbstractVector},
# 		con::VariableBoundConstraint, Z::Traj, inds=1:length(Z)-1)
# 	for (i,k) in enumerate(inds)
# 		vals[i] = con.B*Z[k].z + con.b[k]
# 	end
# end
#
# function jacobian(con::VariableBoundConstraint, z::KnotPoint)
# 	return con.B
# end
#
# function VariableBoundConstraint(n, m, N;
# 		x_max=[Inf*(@SVector ones(n)) for k = 1:N], x_min=[-Inf*(@SVector ones(n)) for k = 1:N],
# 		u_max=[Inf*(@SVector ones(m)) for k = 1:N], u_min=[-Inf*(@SVector ones(m)) for k = 1:N])
# 	@assert length(x_max) == N
# 	@assert length(u_max) == N
# 	@assert length(x_min) == N
# 	@assert length(u_min) == N
#
# 	# Check and convert bounds
# 	for k = 1:N
# 		x_max[k], x_min[k] = checkBounds(Val(n), x_max[k], x_min[k])
# 		u_max[k], u_min[k] = checkBounds(Val(m), u_max[k], u_min[k])
# 	end
#
# 	# Concatenate bounds
# 	z_max = [SVector{n+m}([x_max[k]; u_max[k]]) for k = 1:N]
# 	z_min = [SVector{n+m}([x_min[k]; u_min[k]]) for k = 1:N]
# 	b = [[-z_max[k]; z_min[k]] for k = 1:N]
#
# 	active = map(x->isfinite.(x), b)
# 	equal_active = all(1:N-2) do k
# 		active[k] == active[k+1]
# 	end
# 	if !equal_active
# 		throw(ArgumentError("All bounds must have the same active constraints"))
# 	end
# 	active = active[1]
# 	p = sum(active)
#
# 	inds = SVector{p}(findall(active))
#
# 	b = [bi[inds] for bi in b]
# 	B = SMatrix{2(n+m), n+m}([1.0I(n+m); -1.0I(n+m)])
#
# 	VariableBoundConstraint(n, m, z_max, z_min, b, B[inds,:])
# end



############################################################################################
#  								INDEXED CONSTRAINT 	 									   #
############################################################################################
"""
	IndexedConstraint{C,NM}

Compute a constraint on an arbitrary portion of either the state or control,
or both. Useful for dynamics augmentation. e.g. you are controlling two models, and have
individual constraints on each. You can define constraints as if they applied to the individual
model, and then wrap it in an `IndexedConstraint` to apply it to the appropriate portion of
the concatenated state. Assumes the indexed state portion is contiguous.

Type params:
* S - Inequality or Equality
* W - ConstraintType
* P - Constraint length
* N,M - original state and control dimensions
* NM - N+M
* Bx - location of the first element in the state index
* Bu - location of the first element in the control index
* C - type of original constraint

Constructors:
```julia
IndexedConstraint(n, m, con)
IndexedConstraint(n, m, con, ix::SVector, iu::SVector)
```
where the arguments `n` and `m` are the state and control dimensions of the new dynamics.
`ix` and `iu` are the indices into the state and control vectors. If left out, they are
assumed to start at the beginning of the vector.

NOTE: Only part of this functionality has been tested. Use with caution!
"""
struct IndexedConstraint{C,NM} <: StageConstraint
	n::Int  # new dimension
	m::Int  # new dimension
	n0::Int # old dimension
	m0::Int # old dimension
	con::C
	ix::UnitRange{Int}  # index of old x in new z
	iu::UnitRange{Int}  # index of old u in new z
	iz::SVector{NM,Int} # index of old z in new z
	∇c::Matrix{Float64}
	A::SubArray{Float64,2,Matrix{Float64},Tuple{UnitRange{Int},UnitRange{Int}},false}
	B::SubArray{Float64,2,Matrix{Float64},Tuple{UnitRange{Int},UnitRange{Int}},false}
end

@inline state_dim(con::IndexedConstraint) = con.n
@inline control_dim(con::IndexedConstraint) = con.m
@inline Base.length(con::IndexedConstraint) = length(con.con)
@inline sense(con::IndexedConstraint) = sense(con.con)

function IndexedConstraint(n,m,con::AbstractConstraint,
		ix::UnitRange{Int}, iu::UnitRange{Int})
	p = length(con)
	n0,m0 = length(ix), length(iu)
	iu = iu .+ n
	iz = SVector{n0+m0}([ix; iu])
	w = width(con)
	∇c = zeros(p,w)
	if con isa StageConstraint
		A = view(∇c, 1:p, 1:n0)
		B = view(∇c, 1:p, n0 .+ (1:m0))
	else
		throw(ArgumentError("IndexedConstraint not support for CoupledConstraint yet"))
	end
	IndexedConstraint{typeof(con),n0+m0}(n,m,n0,m0,con,ix,iu,iz,∇c,A,B)
end

function IndexedConstraint(n,m,con::AbstractConstraint)
	if con isa Union{StateConstraint, CoupledStateConstraint}
		m0 = m
	else
		m0 = control_dim(con)
	end
	if con isa Union{ControlConstraint, CoupledControlConstraint}
		n0 = n
	else
		n0 = state_dim(con)
	end
	ix = 1:n0
	iu = 1:m0
	IndexedConstraint(n, m, con, ix, iu)
end

function evaluate(con::IndexedConstraint, z::AbstractKnotPoint)
	z_ = StaticKnotPoint(z, z.z[con.iz])
	evaluate(con.con, z_)
end

@generated function jacobian!(∇c, con::IndexedConstraint{C}, z::AbstractKnotPoint) where C
	if C <: StateConstraint
		assignment = quote
			∇c_ = uview(∇c, :, con.ix)
			isconst = jacobian!(∇c_, con.con, z_)
		end
	elseif C <: ControlConstraint
		assignment = quote
			∇c_ = uview(∇c, :, con.iu)
			isconst = jacobian!(∇c_, con.con, z_)
		end
	else
		assignment = quote
			∇c_ = con.∇c
			isconst = jacobian!(∇c_, con.con, z_)
			uview(∇c, :, con.ix) .= con.A
			uview(∇c, :, con.iu) .= con.B
		end
	end
	quote
		z_ = StaticKnotPoint(z, z.z[con.iz])
		$assignment
		return isconst
	end
end

function change_dimension(con::AbstractConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
	IndexedConstraint(n, m, con, ix, iu)
end
#
# # TODO: define higher-level evaluate! function instead
# @generated function evaluate(con::IndexedConstraint{<:Any,<:Stage,<:Any,N,M}, z::KnotPoint) where {N,M}
# 	ix = SVector{N}(1:N)
# 	iu = N .+ SVector{M}(1:M)
# 	return quote
# 		x0 = state(z)[con.ix]
# 		u0 = control(z)[con.iu]
# 		z_ = StaticKnotPoint([x0; u0], $ix, $iu, z.dt, z.t)
# 		evaluate(con.con, z_)
# 	end
# end
#
# # TODO: define higher-leel jacobian! function instead
# @generated function jacobian!(∇c, con::IndexedConstraint{<:Any,Stage,P,N0,M0},
# 		z::KnotPoint{<:Any,N}) where {P,N0,M0,N}
# 	iP = 1:P
# 	ix = SVector{N0}(1:N0)
# 	iu = SVector{M0}(N0 .+ (1:M0))
# 	if eltype(∇c) <: SizedMatrix
# 		assignment = quote
# 			uview(∇c.data,$iP,iA) .= con.A
# 			uview(∇c.data,$iP,iB) .= con.B
# 		end
# 	else
# 		assignment = quote
# 			uview(∇c,$iP,iA) .= con.A
# 			uview(∇c,$iP,iB) .= con.B
# 		end
# 	end
# 	quote
# 		x0 = state(z)[con.ix]
# 		u0 = control(z)[con.iu]
# 		z_ = StaticKnotPoint([x0;u0], $ix, $iu, z.dt, z.t)
# 		jacobian!(con.∇c, con.con, z_)
# 		iA = con.ix
# 		iB = N .+ con.iu
# 		$assignment
# 	end
# end
#
# @generated function jacobian!(∇c, con::IndexedConstraint{<:Any,State,P,N0,M0},
# 		z::KnotPoint{<:Any,N}) where {P,N0,M0,N}
# 	iP = 1:P
# 	ix = SVector{N0}(1:N0)
# 	iu = SVector{M0}(N0 .+ (1:M0))
# 	if eltype(∇c) <: SizedArray
# 		assignment = :(uview(∇c.data,$iP,iA) .= con.∇c)
# 	else
# 		assignment = :(uview(∇c,$iP,iA) .= con.∇c)
# 	end
# 	quote
# 		x0 = state(z)[con.ix]
# 		u0 = control(z)[con.iu]
# 		z_ = StaticKnotPoint([x0;u0], $ix, $iu, z.dt, z.t)
# 		jacobian!(con.∇c, con.con, z_)
# 		iA = con.ix
# 		$assignment
# 	end
# end
#
# @generated function jacobian!(∇c, con::IndexedConstraint{<:Any,Control,P,N0,M0},
# 		z::KnotPoint{<:Any,N}) where {P,N0,M0,N}
# 	iP = 1:P
# 	ix = SVector{N0}(1:N0)
# 	iu = SVector{M0}(N0 .+ (1:M0))
# 	if eltype(∇c) <: SizedArray
# 		assignment = :(uview(∇c.data,$iP,iB) .= con.∇c)
# 	else
# 		assignment = :(uview(∇c,$iP,iB) .= con.∇c)
# 	end
# 	quote
# 		x0 = state(z)[con.ix]
# 		u0 = control(z)[con.iu]
# 		z_ = StaticKnotPoint([x0;u0], $ix, $iu, z.dt, z.t)
# 		jacobian!(con.∇c, con.con, z_)
# 		iB = con.iu
# 		$assignment
# 	end
# end
