using StaticArrays, ForwardDiff, BenchmarkTools, LinearAlgebra

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


upper_bound(con::AbstractStaticConstraint{Inequality,W,P}) where {P,W} = @SVector zeros(P)
lower_bound(con::AbstractStaticConstraint{Inequality,W,P}) where {P,W} = -Inf*@SVector ones(P)
upper_bound(con::AbstractStaticConstraint{Equality,W,P}) where {P,W} = @SVector zeros(P)
lower_bound(con::AbstractStaticConstraint{Equality,W,P}) where {P,W} = @SVector zeros(P)



struct ConstraintVals{T,W,C,P,NM,PNM}
	con::C
	inds::UnitRange{Int}
	vals::Vector{SVector{P,T}}
	vals_prev::Vector{SVector{P,T}}
	∇c::Vector{SMatrix{P,NM,T,PNM}}
	λ::Vector{SVector{P,T}}
	μ::Vector{SVector{P,T}}
	active::Vector{SVector{P,Bool}}
	c_max::Vector{T}
	function ConstraintVals(con::AbstractStaticConstraint{S, W},
			inds::UnitRange{Int}, vals::V, vals_prev,
			∇c::Vector{SMatrix{P,NM,T,PNM}}, λ::V, μ::V, active::Vector{SVector{P,Bool}},
			c_max::Vector{T}) where {S,W,T,P,NM,PNM, V}
		new{T,W,typeof(con),P,NM,PNM}(con,inds,vals,vals_prev,∇c,λ,μ,active_c_max)
	end
end

function ConstraintVals(con::C, inds::UnitRange) where C
	n,m,p = size(con)
	P = length(inds)
	λ    = [@SVector zeros(p) for k = 1:P]
	μ    = [@SVector ones(p)  for k = 1:P]
	atv  = [@SVector ones(Bool,p) for k = 1:P]
	vals = [@SVector zeros(p) for k = 1:P]
	∇c   = [@SMatrix zeros(p,n+m) for k = 1:P]
	ConstraintVals(con, inds, vals, deepcopy(vals), ∇c, λ, μ, atv, zeros(P))
end

function _index(con::ConstraintVals, k::Int)
	if k ∈ con.inds
		return k - con.inds[1] + 1
	else
		return 0
	end
end


Base.length(::ConstraintVals{T,W,C,P}) where {T,W,C,P} = P
Base.length(con::ConstraintVals, k::Int) = k ∈ con.inds ? length(con) : 0
Base.size(con::ConstraintVals) = size(con.con)
constraint_type(con::ConstraintVals{T,W,C}) where {T,W,C} = C
is_bound(con::ConstraintVals) = is_bound(con.con)
duals(con::ConstraintVals) = con.λ
duals(con::ConstraintVals, k::Int) = con.λ[_index(con,k)]
penalty(con::ConstraintVals) = con.μ
penalty(con::ConstraintVals, k::Int) = con.μ[_index(con,k)]
penalty_matrix(con::ConstraintVals, i::Int) = Diagonal(con.active[i] .* con.μ[i])
lower_bound(con::ConstraintVals) = lower_bound(con.con)
upper_bound(con::ConstraintVals) = upper_bound(con.con)

function evaluate(con::ConstraintVals{T,Stage}, Z::Traj) where T
	for i in eachindex(con.vals)
		k = con.inds[i]
		con.vals[i] = evaluate(con.con, state(Z[k]), control(Z[k]))
	end
end

function evaluate(con::ConstraintVals{T,Dynamical}, Z::Traj) where T
	for i in eachindex(con.vals)
		k = con.inds[i]
		con.vals[i] = evaluate(con.con, state(Z[k+1]), state(Z[k]), control(Z[k]))
	end
end

function jacobian(con::ConstraintVals{T,Stage}, Z::Traj) where T
	for i in eachindex(con.vals)
		k = con.inds[i]
		con.∇c[i] = jacobian(con.con, Z[k])
	end
end

function jacobian(con::ConstraintVals{T,Coupled}, Z::Traj) where T
	for i in eachindex(con.vals)
		k = con.inds[i]
		con.∇c[i] = jacobian(con.con, Z[k+1], Z[k])
	end
end

function update_active_set!(con::ConstraintVals{T,W,C}, tol=0.0) where
		{T,W,C<:AbstractConstraint{Inequality}}
	for i in eachindex(con.vals)
		con.active[i] = @. (con.vals[i] >= tol) | (con.λ[i] > 0)
	end
	return nothing
end

update_active_set!(con::ConstraintVals{T,W,C}, tol=0.0) where
	{T,W,C<:AbstractConstraint{Equality}} = nothing

function update_active_set!(con::ConstraintVals{T,W,C}, ::Val{tol}) where
		{T,W,C<:AbstractConstraint{Inequality},tol}
	for i in eachindex(con.vals)
		con.active[i] = @. (con.vals[i] >= tol) | (con.λ[i] > 0)
	end
	return nothing
end

update_active_set!(con::ConstraintVals{T,W,C}, ::Val{tol}) where
	{T,W,C<:AbstractConstraint{Equality},tol} = nothing

function viol_ineq(v::T, a)::T where T
	for i in eachindex(a)
		v = max(v, max(a[i], 0.0))
	end
	return v
end

function viol_eq(v::T, a)::T where T
	for i in eachindex(a)
		v = max(v, abs(a[i]))
	end
	return v
end

function max_violation!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractConstraint{Inequality}}
	for i in eachindex(con.c_max)
		con.c_max[i] = viol_ineq(0.0, con.vals[i])
	end
	return nothing
end

function max_violation!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractConstraint{Equality}}
	for i in eachindex(con.c_max)
		# con.c_max[i] = norm(con.vals[i],Inf)
		con.c_max[i] = viol_eq(0.0, con.vals[i])
	end
	return nothing
end

get_c_max(con::ConstraintVals) = maximum(con.c_max)

function cost!(J, con::ConstraintVals, Z)
	for (i,k) in enumerate(con.inds)
		c = con.vals[i]
		λ = con.λ[i]
		Iμ = penalty_matrix(con, i)
		J[k] += λ'c + 0.5*c'Iμ*c
	end
end


"""
Assumes constraints, active set, and constrint jacobian have all been calculated
"""
function cost_expansion(E, con::ConstraintVals{T,Stage}, Z)
	ix,iu = Z[1]._x, Z[1]._u
	@inbounds for i in eachindex(con.inds)
		k = con.inds[i]
		c = con.vals[i]
		λ = con.λ[i]
		μ = con.μ[i]
		a = con.active[i]
		Iμ = Diagonal( a .* μ )
		cx = con.∇c[i][:,ix]
		cu = con.∇c[i][:,iu]

		E.xx[k] += cx'Iμ*cx
		E.uu[k] += cu'Iμ*cu
		E.ux[k] += cu'Iμ*cx

		g = Iμ*c + λ
		E.x[k] += cx'g
		E.u[k] += cu'g
	end
end

function dual_update!(con::ConstraintVals{T,W,C},
		opts::AugmentedLagrangianSolverOptions{T}) where
		{T,W,C<:AbstractConstraint{Equality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], -opts.dual_max, opts.dual_max)
	end
end

function dual_update!(con::ConstraintVals{T,P,NM,PNM,C},
		opts::AugmentedLagrangianSolverOptions{T}) where
		{T,P,NM,PNM,C<:AbstractConstraint{Inequality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], 0.0, opts.dual_max)
	end
end

function penalty_update!(con::ConstraintVals{T}, opts::AugmentedLagrangianSolverOptions{T}) where T
	ϕ = opts.penalty_scaling
	μ = con.μ
	for i in eachindex(con.inds)
		μ[i] = clamp.(ϕ * μ[i], 0.0, opts.penalty_max)
	end
end

function reset!(con::ConstraintVals{T,P}, opts::AugmentedLagrangianSolverOptions{T}) where {T,P}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		μ[i] = opts.penalty_initial * @SVector ones(T,P)
		c[i] *= 0.0
		λ[i] *= 0.0
	end
end


struct ConstraintVals{T,P,NM,PNM,C}
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


############################################################################################
#  								CONSTRAINT SETS 										   #
############################################################################################

struct ConstraintSets{T}
	constraints::Vector{<:ConstraintVals}
	p::Vector{Int}
	c_max::Vector{T}
end

function ConstraintSets(N)
	constraints = Vector{ConstraintVals}()
	p = zeros(Int,N)
	c_max = zeros(N)
	ConstraintSets(constraints,p,c_max)
end


Base.length(conSet::ConstraintSets, k) = constraints.p[k]

function ConstraintSets(constraints, N)
	p = zeros(Int,N)
	c_max = zeros(length(constraints))
	for con in constraints
		for k = 1:N
			p[k] += length(con, k)
		end
	end
	ConstraintSets(constraints, p, c_max)
end

function num_constraints!(conSet::ConstraintSets)
	p = conSet.p
	p .*= 0
	for con in conSet.constraints
		for k = 1:length(p)
			p[k] += length(con, k)
		end
	end
end

function max_violation!(conSet::ConstraintSets{T}) where T
	for i in eachindex(conSet.constraints)
		con = conSet.constraints[i]
		max_violation!(con)
		conSet.c_max[i] = maximum(con.c_max::Vector{T})
	end
end

function evaluate(conSet::ConstraintSets, Z::Traj)
	for con in conSet.constraints
		evaluate(con, Z)
	end
end

function jacobian(conSet::ConstraintSets, Z::Traj)
	for con in conSet.constraints
		jacobian(con, Z)
	end
end

function update_active_set!(conSet::ConstraintSets, Z::Traj, tol=0.0)
	for con in conSet.constraints
		update_active_set!(con, tol)
	end
end

Base.iterate(conSet::ConstraintSets) = @inbounds (conSet.constraints[1], 1)
Base.iterate(conSet::ConstraintSets, i) = @inbounds i >= length(conSet.constraints) ? nothing : (conSet.constraints[i+1], i+1)

function reset!(conSet::ConstraintSets, opts)
	for con in conSet.constraints
		reset!(con, opts)
	end
end




############################################################################################
#                              CUSTOM CONSTRAINTS 										   #
############################################################################################

struct GoalConstraint{T,N} <: AbstractStaticConstraint{Equality,Stage,N}
	xf::SVector{N,T}
	Ix::Diagonal{T,SVector{N,T}}
	GoalConstraint(xf::SVector{N,T}) where {N,T} = new{T,N}(xf, Diagonal(@SVector ones(N)))
end
size(con::GoalConstraint{T,N}) where {T,N} = (N,0,N)
evaluate(con::GoalConstraint,x,u) = x - con.xf
jacobian(con::GoalConstraint,z::KnotPoint) = con.Ix

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
	-(x[1] - xc).^2 - (x[2] - yc).^2 + r.^2
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


struct NormConstraint{T} <: AbstractStaticConstraint{Equality,Stage,1}
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


struct ImplicitDynamics{T,L,N,NN} <: AbstractStaticConstraint{Equality,Dynamical}
	model::L
	In::SMatrix{N,N,T,NN}
end

function evaluate(con::ImplicitDynamics, x′, x, u)
	dynamics(con.model, x, u) - x′
end

function jacobian(con::ImplicitDynamics, x′, x, u)
	AB = jacobian(con.model, x, u)
	[AB con.In]
end


struct ExplicitDynamics{Q<:QuadratureRule,L} <: AbstractStaticConstraint{Equality,Coupled}
	model::L
end

function evaluate(con::ExplicitDynamics{HermiteSimpson}, Z2, Z1)
	x′, x = state(Z2), state(Z1)
	u′, u = control(Z2), control(Z1)
	dt = Z1.dt
	xm = (x′ + x)/2 + dt/8*(fVal[k] - fVal[k+1])


function Base.size(con::Union{ImplicitDynamics,ExplicitDynamics})
	n,m = con.model.n, con.model.m
	return n,m,n
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
