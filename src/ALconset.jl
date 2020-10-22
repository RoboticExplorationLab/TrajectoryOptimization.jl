
"""
An [`AbstractConstraintSet`](@ref) that stores the constraint values as well as Lagrange
multiplier and penalty terms for each constraint.

The cost associated with constraint terms in the augmented Lagrangian can be evaluated for

	cost!(J::Vector, ::ALConstraintSet)

which adds the cost at each time step to the vector `J` of length `N`.

The cost expansion for these terms is evaluated along the trajectory `Z` using

	cost_expansion!(E::Objective, conSet::ALConstraintSet, Z)

which also adds the expansion terms to the terms in `E`.

The penalty and multiplier terms can be updated using

	penalty_update!(::ALConstraintSet)
	dual_update!(::ALConstraintSet)

The current set of active constraint (with tolerance `tol`) can be re-calculated using

	update_active_set!(::ALConstraintSet, ::Val{tol})

The maximum penalty can be queried using `max_penalty(::ALConstraintSet)`, and the
penalties and/or multipliers can be reset using

	reset!(::ALConstraintSet)
	reset_penalties!(::ALConstraintSet)
	reset_duals!(::ALConstraintSet)

# Constructor
	ALConstraintSet(::ConstraintList, ::AbstractModel)
	ALConstraintSet(::Problem)
"""
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
Base.iterate(conSet::AbstractConstraintSet) =
	isempty(get_convals(conSet)) ? nothing : (get_convals(conSet)[1].con,1)
Base.iterate(conSet::AbstractConstraintSet, state::Int) =
	state >= length(conSet) ? nothing : (get_convals(conSet)[state+1].con, state+1)
@inline Base.length(conSet) = length(get_convals(conSet))
Base.IteratorSize(::AbstractConstraintSet) = Base.HasLength()
Base.IteratorEltype(::AbstractConstraintSet) = Base.HasEltype()
Base.eltype(::AbstractConstraintSet) = AbstractConstraint

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


# Augmented Lagrangian Updates
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

function cost_expansion!(E::Objective, conSet::ALConstraintSet, Z::AbstractTrajectory,
		init::Bool=false, rezero::Bool=false)
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

"""
	max_penalty(conSet::ALConstraintSet)

Calculate the maximum constrained penalty across all constraints.
"""
function max_penalty(conSet::ALConstraintSet)
	max_penalty!(conSet)
	maximum(conSet.μ_max)
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

# Reset
function reset!(conSet::ALConstraintSet)
    reset_duals!(conSet)
    reset_penalties!(conSet)
end

function reset_duals!(conSet::ALConstraintSet)
	function _reset!(V::Vector{<:SVector})
	    for i in eachindex(V)
	        V[i] = zero(V[i])
	    end
	end
    for i in eachindex(conSet.λ)
        _reset!(conSet.λ[i])
    end
end

function reset_penalties!(conSet::ALConstraintSet)
	function _reset!(V::Vector{<:SVector}, params::ConstraintParams)
	    for i in eachindex(V)
	        V[i] = zero(V[i]) .+ params.μ0
	    end
	end
    for i in eachindex(conSet.μ)
        # reset!(conSet.μ[i], conSet.params[i].μ0)
		# μ0 = conSet.params[i].μ0
        _reset!(conSet.μ[i], conSet.params[i])
    end
end

function shift_fill!(conSet::ALConstraintSet, n=1)
	for i = 1:length(conSet)
		shift_fill!(conSet.convals[i], n)
	end
end