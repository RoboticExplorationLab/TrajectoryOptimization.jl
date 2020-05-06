

mutable struct ConstraintParams{T}
	ϕ::T  	    # penalty scaling parameter
	μ0::T 	    # initial penalty parameter
	μ_max::T    # max penalty parameter
	λ_max::T    # max Lagrange multiplier
end

function ConstraintParams(ϕ::T1 = 10, μ0::T2 = 1.0, μ_max::T3 = 1e8, λ_max::T4 = 1e8) where {T1,T2,T3,T4}
	T = promote_type(T1,T2,T3,T4)
	ConstraintParams(T(ϕ), T(μ0), T(μ_max), T(λ_max))
end

struct ALConstraintSet{T} <: AbstractConstraintSet
    convals::Vector{ConVal}
    errvals::Vector{ConVal}
    λ::Vector{<:Vector}
    μ::Vector{<:Vector}
    active::Vector{<:Vector}
    c_max::Vector{T}
    μ_max::Vector{T}
    μ_maxes::Vector{Vector{T}}
	params::Vector{ConstraintParams{T}}
	p::Vector{Int}
end

function ALConstraintSet(cons::ConstraintList, model::AbstractModel)
    n,m = cons.n, cons.m
    n̄ = RobotDynamics.state_diff_size(model)
    ncon = length(cons)
    useG = model isa LieGroupModel
    errvals = map(1:ncon) do i
        C,c = gen_convals(n̄, m, cons[i], cons.inds[i])
        ConVal(n̄, m, cons[i], cons.inds[i], C, c, useG)
    end
    convals = map(errvals) do errval
        ConVal(n, m, errval)
    end
	errvals = convert(Vector{ConVal}, errvals)
	convals = convert(Vector{ConVal}, convals)
    λ = map(1:ncon) do i
        p = length(cons[i])
        [@SVector zeros(p) for i in cons.inds[i]]
    end
    μ = map(1:ncon) do i
        p = length(cons[i])
        [@SVector ones(p) for i in cons.inds[i]]
    end
    a = map(1:ncon) do i
        p = length(cons[i])
        [@SVector ones(Bool,p) for i in cons.inds[i]]
    end
    c_max = zeros(ncon)
    μ_max = zeros(ncon)
    μ_maxes = [zeros(length(ind)) for ind in cons.inds]
	params = [ConstraintParams() for con in cons.constraints]
    ALConstraintSet(convals, errvals, λ, μ, a, c_max, μ_max, μ_maxes, params, copy(cons.p))
end

@inline ALConstraintSet(prob::Problem) = ALConstraintSet(prob.constraints, prob.model)

# Iteration
Base.iterate(conSet::ALConstraintSet) =
	isempty(get_convals(conSet)) ? nothing : (get_convals(conSet)[1].con,1)
Base.iterate(conSet::ALConstraintSet, state::Int) =
	state >= length(conSet) ? nothing : (get_convals(conSet)[state+1].con, state+1)
@inline Base.length(conSet) = length(get_convals(conSet))
Base.IteratorSize(::ALConstraintSet) = Base.HasLength()
Base.IteratorEltype(::ALConstraintSet) = Base.HasEltype()
Base.eltype(::ALConstraintSet) = AbstractConstraint

"""
	link_constraints!(set1, set2)

Link any common constraints between `set1` and `set2` by setting elements in `set1` to point
to elements in `set2`
"""
function link_constraints!(set1::ALConstraintSet, set2::ALConstraintSet)
	# Find common constraints
	links = Tuple{Int,Int}[]
	for (i,con1) in enumerate(set1)
		for (j,con2) in enumerate(set2)
			if con1 === con2
				push!(links, (i,j))
			end
		end
	end

	# Link values
	for (i,j) in links
		set1.convals[i] = set2.convals[j]
		set1.errvals[i] = set2.errvals[j]
		set1.active[i] = set2.active[j]
		set1.λ[i] = set2.λ[j]
		set1.μ[i] = set2.μ[j]
	end
	return links
end


@inline get_convals(conSet::ALConstraintSet) = conSet.convals
@inline get_errvals(conSet::ALConstraintSet) = conSet.errvals

# Constraint Evaluation
function evaluate!(conSet::AbstractConstraintSet, Z::Traj)
    for conval in get_convals(conSet)
        evaluate!(conval, Z)
    end
end

function jacobian!(conSet::AbstractConstraintSet, Z::Traj)
    for conval in get_convals(conSet)
        jacobian!(conval, Z)
    end
end

function error_expansion!(conSet::AbstractConstraintSet, model::AbstractModel, G)
	@assert get_convals(conSet) == get_errvals(conSet)
	return nothing
end

function error_expansion!(conSet::AbstractConstraintSet, model::LieGroupModel, G)
	convals = get_convals(conSet)
	errvals = get_errvals(conSet)
	for i in eachindex(errvals)
		error_expansion!(errvals[i], convals[i], model, G)
	end
end

# Max values
function max_violation(conSet::AbstractConstraintSet)
	max_violation!(conSet)
    return maximum(conSet.c_max)
end

function max_violation!(conSet::AbstractConstraintSet)
	convals = get_convals(conSet)
	T = eltype(conSet.c_max)
    for i in eachindex(convals)
        max_violation!(convals[i])
        conSet.c_max[i] = maximum(convals[i].c_max::Vector{T})
    end
	return nothing
end

function norm_violation(conSet::AbstractConstraintSet, p=2)
	norm_violation!(conSet, p)
	norm(conSet.c_max, p)
end

function norm_violation!(conSet::AbstractConstraintSet, p=2)
	convals = get_convals(conSet)
	T = eltype(conSet.c_max)
	for i in eachindex(convals)
		norm_violation!(convals[i], p)
		c_max = convals[i].c_max::Vector{T}
		conSet.c_max[i] = norm(c_max, p)
	end
end

function norm_dgrad(conSet::AbstractConstraintSet, dx::Traj, p=1)
	convals = get_convals(conSet)
	T = eltype(conSet.c_max)
	for i in eachindex(convals)
		norm_dgrad!(convals[i], dx, p)
		c_max = convals[i].c_max::Vector{T}
		conSet.c_max[i] = sum(c_max)
	end
	return sum(conSet.c_max)
end

function max_penalty!(conSet::ALConstraintSet{T}) where T
    conSet.c_max .*= 0
    for i in eachindex(conSet.μ)
        maxes = conSet.μ_maxes[i]::Vector{T}
        max_penalty!(maxes, conSet.μ[i])
        conSet.μ_max[i] = maximum(maxes)
    end
end

function max_penalty!(μ_max::Vector{<:Real}, μ::Vector{<:StaticVector})
    for i in eachindex(μ)
        μ_max[i] = maximum(μ[i])
    end
    return nothing
end

function findmax_violation(conSet::AbstractConstraintSet)
	max_violation!(conSet)
	c_max0, j_con = findmax(conSet.c_max) # which constraint
	if c_max0 < eps()
		return "No constraints violated"
	end
	convals = get_convals(conSet)
	conval = convals[j_con]
	i_con = findmax(conval.c_max)[2]  # whicn index
	k_con = conval.inds[i_con] # time step
	con_sense = sense(conval.con)
	viol = violation(con_sense, conval.vals[i_con])
	c_max, i_max = findmax(viol)  # index into constraint
	@assert c_max == c_max0
	con_name = string(typeof(conval.con).name)
	return con_name * " at time step $k_con at " * con_label(conval.con, i_max)
end

# Reset
function reset!(conSet::ALConstraintSet)
    reset_duals!(conSet)
    reset_penalties!(conSet)
end

function reset_duals!(conSet::ALConstraintSet)
    for i in eachindex(conSet.λ)
        reset!(conSet.λ[i], 0.0)
    end
end

function reset_penalties!(conSet::ALConstraintSet)
    for i in eachindex(conSet.μ)
        reset!(conSet.μ[i], conSet.params[i].μ0)
    end
end

function reset!(V::Vector{<:SVector}, v0)
    for i in eachindex(V)
        V[i] = zero(V[i]) .+ v0
    end
end

# Augmented Lagrangian Updated
function dual_update!(conSet::ALConstraintSet)
    for i in eachindex(conSet.λ)
        dual_update!(conSet.convals[i], conSet.λ[i], conSet.μ[i], conSet.params[i])
    end
end

function dual_update!(conval::ConVal, λ::Vector{<:SVector}, μ::Vector{<:SVector}, params::ConstraintParams)
    c = conval.vals
	λ_max = params.λ_max
	λ_min = sense(conval.con) == Equality() ? -λ_max : zero(λ_max)
	for i in eachindex(conval.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], λ_min, λ_max)
	end
end

function penalty_update!(conSet::ALConstraintSet)
	for i in eachindex(conSet.μ)
		penalty_update!(conSet.μ[i], conSet.params[i])
	end
end

function penalty_update!(μ::Vector{<:SVector}, params::ConstraintParams)
	ϕ = params.ϕ
	μ_max = params.μ_max
	for i in eachindex(μ)
		μ[i] = clamp.(ϕ * μ[i], 0.0, μ_max)
	end
end

# Active Set
function update_active_set!(conSet::ALConstraintSet, val::Val{tol}=Val(0.0)) where tol
	for i in eachindex(conSet.active)
		update_active_set!(conSet.active[i], conSet.λ[i], conSet.convals[i], val)
	end
end

function update_active_set!(a::Vector{<:StaticVector}, λ::Vector{<:StaticVector},
		conval::ConVal, ::Val{tol}) where tol
	if sense(conval.con) == Inequality()
		for i in eachindex(a)
			a[i] = @. (conval.vals[i] >= -tol) | (λ[i] > zero(tol))
		end
	end
end

# Cost
function cost!(J::Vector{<:Real}, conSet::ALConstraintSet)
	for i in eachindex(conSet.convals)
		cost!(J, conSet.convals[i], conSet.λ[i], conSet.μ[i], conSet.active[i])
	end
end

function cost!(J::Vector{<:Real}, conval::ConVal, λ::Vector{<:StaticVector},
		μ::Vector{<:StaticVector}, a::Vector{<:StaticVector})
	for (i,k) in enumerate(conval.inds)
		c = SVector(conval.vals[i])
		Iμ = Diagonal(SVector(μ[i] .* a[i]))
		J[k] += λ[i]'c .+ 0.5*c'Iμ*c
	end
end

function cost_expansion!(E::Objective, conSet::ALConstraintSet, Z::Traj, init::Bool=false)
	for i in eachindex(conSet.errvals)
		cost_expansion!(E, conSet.convals[i], conSet.λ[i], conSet.μ[i], conSet.active[i])
	end
end

@generated function cost_expansion!(E::QuadraticObjective{n,m}, conval::ConVal{C}, λ, μ, a) where {n,m,C}
	if C <: StateConstraint
		expansion = quote
			cx = ∇c
			E[k].Q .+= cx'Iμ*cx
			E[k].q .+= cx'g
		end
	elseif C <: ControlConstraint
		expansion = quote
			cu = ∇c
			E[k].R .+= cu'Iμ*cu
			E[k].r .+= cu'g
		end
	elseif C<: StageConstraint
		ix = SVector{n}(1:n)
		iu = SVector{m}(n .+ (1:m))
		expansion = quote
			cx = ∇c[:,$ix]
			cu = ∇c[:,$iu]
			E[k].Q .+= cx'Iμ*cx
			E[k].q .+= cx'g
			E[k].H .+= cu'Iμ*cx
			E[k].R .+= cu'Iμ*cu
			E[k].r .+= cu'g
		end
	else
		throw(ArgumentError("cost expansion not supported for CoupledConstraints"))
	end
	quote
		for (i,k) in enumerate(conval.inds)
			∇c = SMatrix(conval.jac[i])
			c = conval.vals[i]
			Iμ = Diagonal(a[i] .* μ[i])
			g = Iμ*c .+ λ[i]

			$expansion
		end
	end
end
