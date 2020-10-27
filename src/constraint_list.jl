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

"""
	ConstraintList

Stores the set of constraints included in a trajectory optimization problem. Includes a list
of both the constraint types [`AbstractConstraint`](@ref) as well as the knot points at which
the constraint is applied. Each constraint is assumed to apply to a contiguous set of knot points.

A `ConstraintList` supports iteration and indexing over the `AbstractConstraint`s, and
iteration of both the constraints and the indices of the knot points at which they apply
via `zip(cons::ConstraintList)`.

Constraints are added via the [`add_constraint!`](@ref) method, which verifies that the constraint
dimension is consistent with the state and control dimensions of the problem.

The total number of constraints at each knot point can be queried using the
[`num_constraints`](@ref) method.

The constraint list can also be sorted to separate `StageConstraint`s and `CoupledConstraint`s
via the `sort!` method.

A constraint list can be queried if it has a `DynamicsConstraint` via
`has_dynamics_constraint(::ConstraintList)`.

# Constructor
	ConstraintList(n::Int, m::Int, N::Int)
"""
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

"""
	add_constraint!(cons::ConstraintList, con::AbstractConstraint, inds::UnitRange, [idx])

Add constraint `cons` to `ConstraintList` `cons` for knot points given by `inds`.

Use `idx` to determine the location of the constraint in the constraint list.
`idx=-1` (default) adds the constraint at the end of the list.

# Example
Here is an example of adding a goal and control limit constraint for a cartpole swing-up.
```julia
# Dimensions of our problem
n,m,N = 4,1,51    # 51 knot points

# Create our list of constraints
cons = ConstraintList(n,m,N)

# Create the goal constraint
xf = [0,π,0,0]
goalcon = GoalConstraint(xf)
add_constraint!(cons, goalcon, N)  # add to the last time step

# Create control limits
ubnd = 3
bnd = BoundConstraint(n,m, u_min=-ubnd, u_max=ubnd, idx=1)  # make it the first constraint
add_constraint!(cons, bnd, 1:N-1)  # add to all but the last time step

# Indexing
cons[1] === bnd                            # (true)
cons[2] === goal                           # (true)
allcons = [con for con in cons]
cons_and_inds = [(con,ind) in zip(cons)]
cons_and_inds[1] == (bnd,1:n-1)            # (true)
```
"""
function add_constraint!(cons::ConstraintList, con::AbstractConstraint, inds::UnitRange{Int}, idx=-1)
	@assert check_dims(con, cons.n, cons.m) "New constraint not consistent with n=$(cons.n) and m=$(cons.m)"
	@assert inds[end] <= length(cons.p) "Invalid inds, inds[end] must be less than number of knotpoints, $(length(cons.p))"
	if isempty(cons)
		idx = -1
	end
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
Base.iterate(cons::ConstraintList, i::Int) = i < length(cons) ? (cons[i+1], i+1) : nothing
@inline Base.length(cons::ConstraintList) = length(cons.constraints)
Base.IteratorSize(::ConstraintList) = Base.HasLength()
Base.IteratorEltype(::ConstraintList) = Base.HasEltype()
Base.eltype(::ConstraintList) = AbstractConstraint
Base.firstindex(::ConstraintList) = 1
Base.lastindex(cons::ConstraintList) = length(cons.constraints)
Base.keys(cons::ConstraintList) = 1:length(cons)

Base.zip(cons::ConstraintList) = zip(cons.inds, cons.constraints)

@inline Base.getindex(cons::ConstraintList, i::Int) = cons.constraints[i]
Base.getindex(cons::ConstraintList, I) = cons.constraints[I]

for method in (:deepcopy, :copy)
	@eval function Base.$method(cons::ConstraintList)
		cons2 = ConstraintList(cons.n, cons.m, length(cons.p))
		for i in eachindex(cons.constraints)
			con_ = $(method == :deepcopy ? :(copy(cons.constraints[i])) : :(cons.constraints[i]))
			add_constraint!(cons2, con_, copy(cons.inds[i]))
		end
		return cons2
	end
end


"""
	num_constraints(::ConstraintList)
	num_constraints(::Problem)
	num_constraints(::TrajOptNLP)

Return a vector of length `N` constaining the total number of constraint values at each
knot point.
"""
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



############################################################################################
#                                 NLP FUNCTIONS
############################################################################################
"""
	primal_bounds!(zL, zU, cons::ConstraintList; remove=true)

Get the lower and upper bounds on the primal variables imposed by the constraints in `cons`,
where `zL` and `zU` are vectors of length `NN`, where `NN` is the total number of primal
variables in the problem. Returns the modified lower bound `zL` and upper bound `zU`.

If any of the bound constraints are redundant, the strictest bound is returned.

If `remove = true`, these constraints will be removed from `cons`.
"""
function primal_bounds!(zL, zU, cons::ConstraintList, remove::Bool=true)
	NN = length(zL)
	n,m = cons.n, cons.m
	isequal = NN % (n+m) == 0
	N = isequal ? Int(NN / (n+m)) : Int((NN+m)/(n+m))
	rm_inds = Int[]
	for (j,(inds,con)) in enumerate(zip(cons))
		is_bound = false
		for (i,k) in enumerate(inds)
			off = (k-1)*(n+m)
			if !isequal && k == N  # don't allow the indexing to go out of bounds at the last time step
				zind = off .+ (1:n)
			else
				zind = off .+ (1:n+m)
			end
			is_bound |= primal_bounds!(view(zL, zind), view(zU, zind), con)
		end
		is_bound && push!(rm_inds,j)
	end
	if remove
		deleteat!(cons.constraints, rm_inds)
		deleteat!(cons.inds, rm_inds)
	end
	return zL, zU
end

"""
	JacobianStructure

Contains information describing the structure (sparsity) of the constraint Jacobian.
"""
struct JacobianStructure
	NN::Int  # primal variables
	P::Int   # dual variables (constraints)
	nD::Int  # non-zero entries in constraint Jacobian
	cinds::Vector{Vector{UnitRange{Int}}}  # indices into the constraint vector
	zinds::Vector{Matrix{UnitRange{Int}}}  # indices into primal vector
	linds::Vector{Matrix{UnitRange{Int}}}  # indices into Jacobian vector
end

function JacobianStructure(cons::ConstraintList, structure=:by_knotpoint)
	n,m = cons.n, cons.m
    N = length(cons.p)
	isequal = false
	if has_dynamics_constraint(cons)
		isequal = integration(cons[end]) <: Implicit
	end
	NN = num_vars(cons.n, cons.m, N, isequal)
	P = sum(num_constraints(cons))
    numcon = length(cons.constraints)

	zinds_ = gen_zinds(n, m, N, isequal)
	cinds = [[1:0 for j in eachindex(cons.inds[i])] for i in 1:length(cons)]
	zinds = [[1:0 for j in eachindex(cons.inds[i]), k in widths(cons[i],n,m)] for i in 1:length(cons)]
	linds = [[1:0 for j in eachindex(cons.inds[i]), k in widths(cons[i],n,m)] for i in 1:length(cons)]


    # Dynamics and general constraints
    idx = 0
	nD = 0
	if structure == :by_constraint
	    for (i,con) in enumerate(cons.constraints)
			len = length(con)
			zs = get_inds(con)
			for (j,k) in enumerate(cons.inds[i])
				inds = idx .+ (1:len)
				cinds[i][j] = inds
				idx += len
				off_z = (k-1)*(n+m)
				for l in length(zs)
					zi = zs[l] .+ off_z       # indices of depedent primal variables
					blk_len = len*length(zi)  # number of elements in the Jacobian block
					zinds[i][j,l] = zi
					linds[i][j,l] = nD .+ (1:blk_len)
					nD += blk_len
				end
	        end
	    end
	elseif structure == :by_knotpoint
		for k = 1:N
			for (i,con) in enumerate(cons)
				inds = cons.inds[i]
				len = length(con)
				zs = get_inds(con,n,m)
				if k in inds
					j = k - inds[1] + 1
					cinds[i][j] = idx .+ (1:len)
					idx += len

					off_z = (k-1)*(n+m)
					for l in 1:length(zs)
						zi = zs[l] .+ off_z       # indices of depedent primal variables
						blk_len = len*length(zi)  # number of elements in the Jacobian block
						zinds[i][j,l] = zi
						linds[i][j,l] = nD .+ (1:blk_len)
						nD += blk_len
					end
				end
			end
		end
	end
	return JacobianStructure(NN, P, nD, cinds, zinds, linds)
end

function jacobian_structure!(D::AbstractMatrix, jac::JacobianStructure)
	ncons = length(jac.cinds)
	for i = 1:ncons
		cinds = jac.cinds[i]
		zinds = jac.zinds[i]
		linds = jac.linds[i]
		for j = 1:length(cinds)
			for k = 1:size(linds,2)
				D[cinds[j], zinds[j,k]] = linds[j,k]
			end
		end
	end
	return D
end

@inline jacobian_structure(jac::JacobianStructure) =
	jacobian_structure!(spzeros(Int,jac.P, jac.NN), jac)

function gen_convals(Dv::AbstractVector, d::AbstractVector, cons::ConstraintList, jac=JacobianStructure(cons))
	n,m = cons.n, cons.m  # TODO: account for state diff size
	C = map(enumerate(zip(cons))) do (i,(inds,con))
		len = length(con)
		ws = widths(con, n, m)
		[reshape(view(Dv, jac.linds[i][j,l]), len, ws[l])
			for j = 1:length(inds), l = 1:length(ws)]
	end
	c = map(enumerate(zip(cons))) do (i,(inds,con))
		[view(d, jac.cinds[i][j]) for j = 1:length(inds)]
	end
	return C,c
end

function gen_convals(D::AbstractMatrix, d::AbstractVector, cons::ConstraintList, jac=JacobianStructure(cons))
	n,m = cons.n, cons.m  # TODO: account for state diff size
	ncons = length(cons)
	C = map(enumerate(zip(cons))) do (i,(inds,con))
		len = length(con)
		ws = widths(con, n, m)
		[view(D, jac.cinds[i][j], jac.zinds[i][j,l])
			for j = 1:length(inds), l = 1:length(ws)]
	end

	c = map(enumerate(zip(cons))) do (i,(inds,con))
		[view(d, jac.cinds[i][j]) for j = 1:length(inds)]
	end
    return C,c
end

"""
	gen_con_inds(cons::ConstraintList, structure::Symbol)

Generate the indices into the concatenated constraint vector for each constraint.
Determines the bandedness of the Jacobian
"""
function gen_con_inds(conSet::ConstraintList, structure=:by_knotpoint)
    return cinds, linds
end

# function jacobian_structure(cons::ConstraintList)
# 	isequal = false
# 	if has_dynamics_constraint(cons)
# 		isequal = integration(cons[end]) <: Implicit
# 	end
# 	N = length(cons.p)
# 	NN = num_vars(cons.n, cons.m, N, isequal)
# 	P = sum(num_constraints(cons))
# 	D = spzeros(NN,P)
#
# 	idx = 0
# 	for k = 1:N
# 		for (i,conval) in enumerate(get_convals(conSet))
# 			p = length(conval.con)
# 			if k in conval.inds
# 				for (l,w) in enumerate(widths(conval.con))
# 					blk_len = p*w
# 					inds = reshape(idx .+ (1:blk_len), p, w)
# 					j = _index(conval,k)
# 					# linds[i][j] = inds
# 					conval.jac[j,l] .= inds
# 					idx += blk_len
# 				end
# 			end
# 		end
# 	end
# end
