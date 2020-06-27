

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


# Constraint Evaluation
function evaluate!(conSet::AbstractConstraintSet, Z::AbstractTrajectory)
    for conval in get_convals(conSet)
        evaluate!(conval, Z)
    end
end

function jacobian!(conSet::AbstractConstraintSet, Z::AbstractTrajectory)
    for conval in get_convals(conSet)
        jacobian!(conval, Z)
    end
end

function ∇jacobian!(G::Vector{<:Matrix}, conSet::AbstractConstraintSet, Z::AbstractTrajectory,
		λ::Vector{<:Vector})
	for (i,conval) in enumerate(get_convals(conSet))
		∇jacobian!(G[i], conval, Z, λ[i])
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

"""
	norm_violation(conSet, [p=2])
	norm_violation(conVal, [p=2])

Calculate the norm of the violation of a `AbstractConstraintSet`, a `Conval` using the `p`-norm.

	norm_violation(sense::ConstraintSense, v::AbstractVector, [p=2])

Calculate the `p`-norm of constraint violations given the vector of constraint values `v`,
	for a constraint of sense `sense` (either `Inequality()` or `Equality()`).
	Assumes that positive values are violations for inequality constraints.
"""
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

function norm_dgrad(conSet::AbstractConstraintSet, dx::AbstractTrajectory, p=1)
	convals = get_convals(conSet)
	T = eltype(conSet.c_max)
	for i in eachindex(convals)
		norm_dgrad!(convals[i], dx, p)
		c_max = convals[i].c_max::Vector{T}
		conSet.c_max[i] = sum(c_max)
	end
	return sum(conSet.c_max)
end


"""
	findmax_violation(conSet)

Return details on the where the largest violation occurs. Returns a string giving the
constraint type, time step index, and index into the constraint.
"""
function findmax_violation(conSet::AbstractConstraintSet)
	max_violation!(conSet)
	c_max0, j_con = findmax(conSet.c_max) # which constraint
	if c_max0 < eps()
		return "No constraints violated"
	end
	convals = get_convals(conSet)
	conval = convals[j_con]
	i_con = findmax(conval.c_max)[2]  # which index
	k_con = conval.inds[i_con] # time step
	con_sense = sense(conval.con)
	viol = [violation(con_sense, v) for v in conval.vals[i_con]]
	c_max, i_max = findmax(viol)  # index into constraint
	@assert c_max == c_max0
	con_name = string(typeof(conval.con).name)
	return con_name * " at time step $k_con at " * con_label(conval.con, i_max)
end
