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

Base.copy(con::AbstractConstraint) = con

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
	xf::MVector{P,T}
	inds::SVector{P,Int}
	function GoalConstraint(xf::AbstractVector{T}, inds::SVector{p,Int}) where {p,T}
		new{p,T}(length(xf), xf[inds], inds)
	end
	function GoalConstraint(n::Int, xf::MVector{P,T}, inds::SVector{P,Int}) where {P,T}
		new{P,T}(n, xf, inds)
	end
end

function GoalConstraint(xf::AbstractVector, inds=1:length(xf))
	p = length(inds)
	inds = SVector{p}(inds)
	GoalConstraint(xf, inds)
end

Base.copy(con::GoalConstraint) = GoalConstraint(copy(con.xf), con.inds)

@inline sense(::GoalConstraint) = Equality()
@inline Base.length(con::GoalConstraint{P}) where P = P
@inline state_dim(con::GoalConstraint) = con.n
@inline is_bound(::GoalConstraint) = true
function primal_bounds!(zL,zU,con::GoalConstraint)
	for (i,j) in enumerate(con.inds)
		zL[j] = con.xf[i]
		zU[j] = con.xf[i]
	end
	return true
end

evaluate(con::GoalConstraint, x::StaticVector) = x[con.inds] - con.xf
function jacobian!(∇c, con::GoalConstraint, z::KnotPoint)
	T = eltype(∇c)
	for (i,j) in enumerate(con.inds)
		∇c[i,j] = one(T)
	end
	return true
end

∇jacobian!(G, con::GoalConstraint, z::AbstractKnotPoint, λ::AbstractVector) = true # zeros

function change_dimension(con::GoalConstraint, n::Int, m::Int, xi=1:n, ui=1:m)
	GoalConstraint(con.n, con.xf, xi[con.inds])
end

function set_goal_state!(con::GoalConstraint, xf::AbstractVector)
	if length(xf) != length(con.xf)
		for (i,j) in enumerate(con.inds)
			con.xf[i] = xf[j]
		end
	else
		con.xf .= xf
	end
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
	A::SizedMatrix{P,W,T,2,Matrix{T}}
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
	T = promote_type(eltype(A), eltype(b))
	A = SizedMatrix{p,q,T}(A)
	b = SVector{p,T}(b)
	LinearConstraint(n,m, A, b, sense, inds)
end

Base.copy(con::LinearConstraint{S}) where S = 
	LinearConstraint(con.n, con.m, copy(con.A), copy(con.b), S(), con.inds)

@inline sense(con::LinearConstraint) = con.sense
@inline Base.length(con::LinearConstraint{<:Any,P}) where P = P
@inline state_dim(con::LinearConstraint) = con.n
@inline control_dim(con::LinearConstraint) = con.m
evaluate(con::LinearConstraint, z::AbstractKnotPoint) = con.A*z.z[con.inds] .- con.b
function jacobian!(∇c, con::LinearConstraint, z::AbstractKnotPoint)
	∇c[:,con.inds[1]:con.inds[end]] .= con.A
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
	function CircleConstraint{P,T}(n::Int, xc::AbstractVector, yc::AbstractVector, radius::AbstractVector,
			xi=1, yi=2) where {P,T}
    	@assert length(xc) == length(yc) == length(radius) == P "Lengths of xc, yc, and radius must be equal. Got lengths ($(length(xc)), $(length(yc)), $(length(radius)))"
        new{P,T}(n, xc, yc, radius, xi, yi)
    end
end
function CircleConstraint(n::Int, xc::AbstractVector, yc::AbstractVector, radius::AbstractVector,
		xi=1, yi=2)
    T = promote_type(eltype(xc), eltype(yc), eltype(radius))
    P = length(xc)
    CircleConstraint{P,T}(n, xc, yc, radius, xi, yi)
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
	function SphereConstraint{P,T}(n::Int, xc::AbstractVector, yc::AbstractVector,
            zc::AbstractVector, radius::AbstractVector,
			xi=1, yi=2, zi=3) where {P,T}
    	@assert length(xc) == length(yc) == length(radius) == length(zc) == P "Lengths of xc, yc, zc, and radius must be equal. Got lengths ($(length(xc)), $(length(yc)), $(length(zc)), $(length(radius)))"
        new{P,T}(n, xc, yc, zc, xi, yi, zi, radius)
    end
end
function SphereConstraint(n::Int, xc::AbstractVector, yc::AbstractVector,
        zc::AbstractVector, radius::AbstractVector,
		xi=1, yi=2, zi=3)
    T = promote_type(eltype(xc), eltype(yc), eltype(zc), eltype(radius))
    P = length(xc)
    SphereConstraint{P,T}(n, xc, yc, zc, radius, xi, yi, zi)
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

"""
    CollisionConstraint

Enforces a pairwise non self-collision constraint on the state, such that
    `norm(x[x1] - x[x2]).^2 > r^2`,
    where `x1` and `x2` are the indices of the positions of the respective bodies and `r`
    is the collision radius.

# Constructor
CollisionConstraint(n::Int, x1::AbstractVector{Int}, x2::AbstractVector{Int}, r::Real)
"""
struct CollisionConstraint{D} <: StateConstraint
	n::Int
    x1::SVector{D,Int}
    x2::SVector{D,Int}
    radius::Float64
    function CollisionConstraint(n::Int, x1::AbstractVector{Int}, x2::AbstractVector{Int}, r::Real)
        @assert length(x1) == length(x2) "Position dimensions must be of equal length, got $(length(x1)) and $(length(x2))"
        D = length(x1)
        new{D}(n, x1, x2, r)
    end
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
``\\|y\\|_2 \\leq a``
where ``y`` is made up of elements from the state and/or control vectors.
The can be equality constraint, e.g. ``y^T y - a^2 = 0``, an inequality constraint,
where `y^T y - a^2 \\leq 0`, or a second-order constraint.

# Constructor:
```
NormConstraint(n, m, a, sense, [inds])
```
where `n` is the number of states,
    `m` is the number of controls,
    `a` is the constant on the right-hand side of the equation,
    `sense` is `Inequality()`, `Equality()`, or `SecondOrderCone()`, and
    `inds` can be a `UnitRange`, `AbstractVector{Int}`, or either `:state` or `:control`

# Examples:
```julia
NormConstraint(3, 2, 4, Equality(), :control)
```
creates a constraint equivalent to
``\\|u\\|^2 = 16.0`` for a problem with 2 controls.

```julia
NormConstraint(3, 2, 3, Inequality(), :state)
```
creates a constraint equivalent to
``\\|x\\|^2 \\leq 9`` for a problem with 3 states.

```julia
NormConstraint(3, 2, 5, SecondOrderCone(), :control)
```
creates a constraint equivalent to 
``\\|x\\|_2 \\leq 5``.

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
@inline Base.length(::NormConstraint{SecondOrderCone,D}) where D = D + 1

function evaluate(con::NormConstraint, z::AbstractKnotPoint)
	x = z.z[con.inds]
	return @SVector [x'x - con.val*con.val]
end

function evaluate(con::NormConstraint{SecondOrderCone}, z::AbstractKnotPoint)
	v = z.z[con.inds]
	return push(v, con.val)
end

function jacobian!(∇c, con::NormConstraint, z::AbstractKnotPoint)
	x = z.z[con.inds]
	∇c[1,con.inds] .= 2*x
	return false
end

function jacobian!(∇c, con::NormConstraint{SecondOrderCone}, z::AbstractKnotPoint)
	for (i,j) in enumerate(con.inds)
		∇c[i,j] = 1.0 
	end
	return true
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

Base.copy(bnd::BoundConstraint{P,nm,T}) where {P,nm,T} =
	BoundConstraint(bnd.n, bnd.m, bnd.z_max, bnd.z_min, 
		copy(bnd.i_max), copy(bnd.i_min), bnd.inds)

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
checkBounds(n::Int, u::AbstractVector, l::Real) = checkBounds(n, u, fill(l,n))
checkBounds(n::Int, u::Real, l::AbstractVector) = checkBounds(n, fill(u,n), l)


@inline state_dim(con::BoundConstraint) = con.n
@inline control_dim(con::BoundConstraint) = con.m
@inline is_bound(::BoundConstraint) = true
@inline lower_bound(bnd::BoundConstraint) = bnd.z_min
@inline upper_bound(bnd::BoundConstraint) = bnd.z_max
@inline sense(::BoundConstraint) = Inequality()
@inline Base.length(con::BoundConstraint) = length(con.i_max) + length(con.i_min)

function primal_bounds!(zL, zU, bnd::BoundConstraint)
	for i = 1:length(zL)
		zL[i] = max(bnd.z_min[i], zL[i])
		zU[i] = min(bnd.z_max[i], zU[i])
	end
	return true
end

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

∇jacobian!(G, con::BoundConstraint, z::AbstractKnotPoint, λ::AbstractVector) = true # zeros

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
	IndexedConstraint{C,N,M}

Compute a constraint on an arbitrary portion of either the state or control,
or both. Useful for dynamics augmentation. e.g. you are controlling two models, and have
individual constraints on each. You can define constraints as if they applied to the individual
model, and then wrap it in an `IndexedConstraint` to apply it to the appropriate portion of
the concatenated state. Assumes the indexed state or control portion is contiguous.

# Type params:
* S - Inequality or Equality
* W - ConstraintType
* P - Constraint length
* N,M - original state and control dimensions
* NM - N+M
* Bx - location of the first element in the state index
* Bu - location of the first element in the control index
* C - type of original constraint

# Constructors:
```julia
IndexedConstraint(n, m, con)
IndexedConstraint(n, m, con, ix::UnitRange, iu::UnitRange)
```
where the arguments `n` and `m` are the state and control dimensions of the new dynamics.
`ix` and `iu` are the indices into the state and control vectors. If left out, they are
assumed to start at the beginning of the vector.

NOTE: Only part of this functionality has been tested. Use with caution!
"""
struct IndexedConstraint{C,N,M} <: StageConstraint
	n::Int  # new dimension
	m::Int  # new dimension
	n0::Int # old dimension
	m0::Int # old dimension
	con::C
	ix::SVector{N,Int}  # index of old x in new z
	iu::SVector{M,Int}  # index of old u in new z
	∇c::Matrix{Float64}
	A::SubArray{Float64,2,Matrix{Float64},Tuple{UnitRange{Int},UnitRange{Int}},false}
	B::SubArray{Float64,2,Matrix{Float64},Tuple{UnitRange{Int},UnitRange{Int}},false}
end

@inline state_dim(con::IndexedConstraint) = con.n
@inline control_dim(con::IndexedConstraint) = con.m
@inline Base.length(con::IndexedConstraint) = length(con.con)
@inline sense(con::IndexedConstraint) = sense(con.con)

function Base.copy(c::IndexedConstraint{C,n0,m0}) where {C,n0,m0}
	IndexedConstraint{C,n0,m0,}(c.n, c.m, c.n0, c.m0, copy(c.con), c.ix, c.iu,
		copy(∇c),copy(A),copy(B))
end

function IndexedConstraint(n,m,con::AbstractConstraint,
		ix::UnitRange{Int}, iu::UnitRange{Int})
	p = length(con)
	n0,m0 = length(ix), length(iu)
	iu = iu .+ n
	iz = SVector{n0+m0}([ix; iu])
	w = widths(con)[1]
	∇c = zeros(p,w)
	if con isa StageConstraint
		if con isa ControlConstraint
			A = view(∇c, 1:p, 1:0)
			B = view(∇c, 1:p, 1:m0)
		else
			A = view(∇c, 1:p, 1:n0)
			if con isa StateConstraint
				B = view(∇c, 1:p, n0:n0-1)
			else
				B = view(∇c, 1:p, n0 .+ (1:m0))
			end
		end
	else
		throw(ArgumentError("IndexedConstraint not support for CoupledConstraint yet"))
	end
	IndexedConstraint{typeof(con),n0,m0}(n,m,n0,m0,con,ix,iu,∇c,A,B)
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
	x0 = z.z[con.ix]
	u0 = z.z[con.iu]
	z_ = StaticKnotPoint(x0, u0, z.dt, z.t)
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
		x0 = z.z[con.ix]
		u0 = z.z[con.iu]
		z_ = StaticKnotPoint(x0, u0, z.dt, z.t)
		$assignment
		return isconst
	end
end

@inline is_bound(idx::IndexedConstraint) = is_bound(idx.con)
@inline upper_bound(idx::IndexedConstraint) = upper_bound(idx.con)
@inline lower_bound(idx::IndexedConstraint) = lower_bound(idx.con)

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

struct QuatVecEq{T} <: StateConstraint
    n::Int
    qf::UnitQuaternion{T}
    qind::SVector{4,Int}
end
function evaluate(con::QuatVecEq, x::StaticVector)
    qf = Rotations.params(con.qf)
    q = normalize(x[con.qind])
    dq = qf'q
    if dq < 0
        qf *= -1
    end
    return -SA[qf[2] - q[2], qf[3] - q[3], qf[4] - q[4]] 
end
sense(::QuatVecEq) = Equality()
state_dim(con::QuatVecEq) = con.n
Base.length(con::QuatVecEq) = 3