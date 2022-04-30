############################################################################################
#					             CONSTRAINT LIST										   #
############################################################################################
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
dimension is consistent with the state and control dimensions at the knot points at which 
they are applied. 

The total number of constraints at each knot point can be queried using the
[`num_constraints`](@ref) method.

# Constructors

	ConstraintList(nx, nu)
	ConstraintList(n, m, N)
	ConstraintList(models)

Where `nx` and `nu` are `N`-dimensional vectors that specify the state and control dimension 
at each knot point. If these are the same for the entire trajectory, the user can use the 
2nd constructor. Alternatively, they can be constructed automatically from `models`, a 
vector of `DiscreteDynamics` models.
"""
struct ConstraintList <: AbstractConstraintSet
	nx::Vector{Int}
	nu::Vector{Int}
	constraints::Vector{AbstractConstraint}
	inds::Vector{UnitRange{Int}}
	sigs::Vector{FunctionSignature}
	diffs::Vector{DiffMethod}
	p::Vector{Int}
	function ConstraintList(nx::AbstractVector{<:Integer}, nu::AbstractVector{<:Integer})
		N = length(nx)
		constraints = AbstractConstraint[]
		inds = UnitRange{Int}[]
		p = zeros(Int,N)
		sigs = FunctionSignature[]
		diffs = DiffMethod[]
		new(nx, nu, constraints, inds, sigs, diffs, p)
	end
end

function ConstraintList(n::Integer, m::Integer, N::Integer)
	nx = fill(n, N)
	nu = fill(m, N)
	return ConstraintList(nx, nu)
end

function ConstraintList(models::Vector{<:DiscreteDynamics})
	ConstraintList(RD.dims(models)...)
end


"""
	add_constraint!(cons::ConstraintList, con::AbstractConstraint, inds::UnitRange, [idx, sig, diffmethod])

Add constraint `con` to `ConstraintList` `cons` for knot points given by `inds`.

Use `idx` to determine the location of the constraint in the constraint list.
`idx=-1` (default) adds the constraint at the end of the list.

The `FunctionSignature` and `DiffMethod` used to evaluate the constraint can be specified by
the `sig` and `diffmethod` keyword arguments, respectively.

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
function add_constraint!(cons::ConstraintList, con::AbstractConstraint, inds::UnitRange{Int}, 
						 idx=-1; sig::FunctionSignature=RD.default_signature(con), 
						 diffmethod::DiffMethod=RD.default_diffmethod(con)
)
	for (i,k) in enumerate(inds)
		if !check_dims(con, cons.nx[k], cons.nu[k])
			throw(DimensionMismatch("New constraint not consistent with n=$(cons.nx[k]) and m=$(cons.nu[k]) at time step $k."))
		end
	end
	@assert inds[end] <= length(cons.p) "Invalid inds, inds[end] must be less than number of knotpoints, $(length(cons.p))"
	if isempty(cons)
		idx = -1
	end
	if idx == -1
		push!(cons.constraints, con)
		push!(cons.inds, inds)
		push!(cons.diffs, diffmethod)
		push!(cons.sigs, sig)
	elseif 0 < idx <= length(cons)
		insert!(cons.constraints, idx, con)
		insert!(cons.inds, idx, inds)
		insert!(cons.diffs, idx, diffmethod)
		insert!(cons.sigs, idx, sig)
	else
		throw(ArgumentError("cannot insert constraint at index=$idx. Length = $(length(cons))"))
	end
	num_constraints!(cons)
	@assert length(cons.constraints) == length(cons.inds)
end

@inline add_constraint!(cons::ConstraintList, con::AbstractConstraint, k::Int, idx=-1; kwargs...) =
	add_constraint!(cons, con, k:k, idx; kwargs...)

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

RD.state_dim(cons::ConstraintList, k) = cons.nx[k]
RD.control_dim(cons::ConstraintList, k) = cons.nu[k]

"""
	functionsignature(::ConstraintList, i)

Get the `FunctionSignature` used to evaluate the `i`th constraint in the constraint list.
"""
functionsignature(cons::ConstraintList, i::Integer) = cons.sigs[i]

"""
	diffmethod(::ConstraintList, i)

Get the `DiffMethod` used to evaluate the Jacobian for `i`th constraint in the constraint 
list.
"""
diffmethod(cons::ConstraintList, i::Integer) = cons.diffs[i]

"""
	constraintindices(::ConstraintList, i)

Get the knot point indices at which the `i`th constraint is applied.
"""
constraintindices(cons::ConstraintList, i::Integer) = cons.inds[i]

for method in (:deepcopy, :copy)
	@eval function Base.$method(cons::ConstraintList)
		cons2 = ConstraintList(cons.nx, cons.nu)
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

Return a vector of length `N` constaining the total number of constraint values at each
knot point.
"""
@inline num_constraints(cons::ConstraintList) = cons.p

function num_constraints!(cons::ConstraintList)
	cons.p .*= 0
	for i = 1:length(cons)
		p = RD.output_dim(cons[i])
		for k in cons.inds[i]
			cons.p[k] += p
		end
	end
end

function change_dimension(cons::ConstraintList, n::Int, m::Int, ix=1:n, iu=1:m)
	@assert all(x->x == cons.nx[1], cons.nx) "change_dimension not supported when the state dimension changes along the trajectory."
	@assert all(x->x == cons.nu[1], cons.nu) "change_dimension not supported when the control dimension changes along the trajectory."
	new_list = ConstraintList(n, m, length(cons.p))
	for (i,con) in enumerate(cons)
		new_con = change_dimension(con, n, m, ix, iu)
		add_constraint!(new_list, new_con, cons.inds[i])
	end
	return new_list
end