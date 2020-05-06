############################################################################################
#					             CONSTRAINT LIST										   #
############################################################################################
"""
	AbstractConstraintSet

Stores constraint error and Jacobian values, correctly accounting for the error state if
necessary.

# Interface
- `get_convals(::AbstractConstraintSet)::Vector{<:ConVal}` where the size of the Jacobians
	match the full state dimension
- `get_errvals(::AbstractConstraintSet)::Vector{<:ConVal}` where the size of the Jacobians
	match the error state dimension
- must have field `c_max::Vector{<:AbstractFloat}` of length `length(get_convals(conSet))`

# Methods
Once the previous interface is defined, the following methods are defined
- `Base.iterate`: iterates over `get_convals(conSet)`
- `Base.length`: number of independent constraints
- `evaluate!(conSet, Z::Traj)`: evaluate the constraints over the entire trajectory `Z`
- `jacobian!(conSet, Z::Traj)`: evaluate the constraint Jacobians over the entire trajectory `Z`
- `error_expansion!(conSet, model, G)`: evaluate the Jacobians for the error state using the
	state error Jacobian `G`
- `max_violation(conSet)`: return the maximum constraint violation
- `findmax_violation(conSet)`: return details about the location of the maximum
	constraint violation in the trajectory
"""
abstract type AbstractConstraintSet end

struct ConstraintList <: AbstractConstraintSet
	n::Int
	m::Int
	constraints::Vector{AbstractConstraint}
	inds::Vector{UnitRange{Int}}
	p::Vector{Int}
	function ConstraintList(n::Int, m::Int, N::Int)
		constraints = AbstractConstraint[]
		inds = UnitRange{Int}[]
		p = zeros(Int,N)
		new(n, m, constraints, inds, p)
	end
end

function add_constraint!(cons::ConstraintList, con::AbstractConstraint, inds::UnitRange{Int}, idx=-1)
	@assert check_dims(con, cons.n, cons.m) "New constaint not consistent with n=$(cons.n) and m=$(cons.m)"
	@assert inds[end] <= length(cons.p) "Invalid inds, inds[end] must be less than number of knotpoints, $(length(cons.p))"
	if idx == -1
		push!(cons.constraints, con)
		push!(cons.inds, inds)
	elseif 0 < idx <= length(cons)
		insert!(cons.constraints, idx, con)
		insert!(cons.inds, idx, inds)
	else
		throw(ArgumentError("cannot insert constraint at index=$idx. Length = $(length(cons))"))
	end
	num_constraints!(cons)
	@assert length(cons.constraints) == length(cons.inds)
end

@inline add_constraint!(cons::ConstraintList, con::AbstractConstraint, k::Int, idx=-1) =
	add_constraint!(cons, con, k:k, idx)

# Iteration
Base.iterate(cons::ConstraintList) = length(cons) == 0 ? nothing : (cons[1], 1)
Base.iterate(cons::ConstraintList, i) = i < length(cons) ? (cons[i+1], i+1) : nothing
@inline Base.length(cons::ConstraintList) = length(cons.constraints)
Base.IteratorSize(::ConstraintList) = Base.HasLength()
Base.IteratorEltype(::ConstraintList) = Base.HasEltype()
Base.eltype(::ConstraintList) = AbstractConstraint
Base.firstindex(::ConstraintList) = 1
Base.lastindex(cons::ConstraintList) = length(cons.constraints)

Base.zip(cons::ConstraintList) = zip(cons.inds, cons.constraints)

@inline Base.getindex(cons::ConstraintList, i::Int) = cons.constraints[i]

function Base.copy(cons::ConstraintList)
	cons2 = ConstraintList(cons.n, cons.m, length(cons.p))
	for i in eachindex(cons.constraints)
		add_constraint!(cons2, cons.constraints[i], copy(cons.inds[i]))
	end
	return cons2
end

@inline num_constraints(cons::ConstraintList) = cons.p

function num_constraints!(cons::ConstraintList)
	cons.p .*= 0
	for i = 1:length(cons)
		p = length(cons[i])
		for k in cons.inds[i]
			cons.p[k] += p
		end
	end
end

function change_dimension(cons::ConstraintList, n::Int, m::Int, ix=1:n, iu=1:m)
	new_list = ConstraintList(n, m, length(cons.p))
	for (i,con) in enumerate(cons)
		new_con = change_dimension(con, n, m, ix, iu)
		add_constraint!(new_list, new_con, cons.inds[i])
	end
	return new_list
end

# sort the constraint list by stage < coupled, preserving ordering
function Base.sort!(cons::ConstraintList; rev::Bool=false)
	lt(con1,con2) = false
	lt(con1::StageConstraint, con2::CoupledConstraint) = true
	inds = sortperm(cons.constraints, alg=MergeSort, lt=lt, rev=rev)
	permute!(cons.inds, inds)
	permute!(cons.constraints, inds)
	return cons
end

function has_dynamics_constraint(conSet::ConstraintList)
	for con in conSet
		if con isa DynamicsConstraint
			return true
		end
	end
	return false
end
