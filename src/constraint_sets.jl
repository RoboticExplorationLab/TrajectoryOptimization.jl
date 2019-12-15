export
    ConstraintSet


############################################################################################
#  								CONSTRAINT SETS 										   #
############################################################################################

""" $(TYPEDEF) Set of all constraints for a trajectory optimization problem
Holds a vector of [`ConstraintVals`](@ref) that specify where in the trajectory each constraint
is applied. The `ConstraintSet` efficiently dispatches functions to all of the constraints.

# Constructors:```julia
ConstraintSet(n,m,N)
ConstraintSet(n,m,Vector{<:ConstraintVals},N)
```
"""
struct ConstraintSet{T}
	n::Int
	m::Int
	constraints::Vector{ConstraintVals}
	p::Vector{Int}
	c_max::Vector{T}
end

function ConstraintSet(n::Int,m::Int,constraints, N)
	@assert !isempty(constraints)
	p = zeros(Int,N)
	c_max = zeros(length(constraints))
	for con in constraints
		for k = 1:N
			p[k] += length(con, k)
		end
	end
	cons = ConstraintVals[]
	append!(cons, constraints)
	ConstraintSet(n,m, cons, p, c_max)
end

function ConstraintSet(n,m,N)
	constraints = Vector{ConstraintVals}()
	p = zeros(Int,N)
	c_max = zeros(0)
	ConstraintSet(n,m,constraints,p,c_max)
end

"Get size of state and control dimensions"
Base.size(conSet::ConstraintSet) = conSet.n, conSet.m, length(conSet.constraints)
"Get number of separate constraints (i.e. ConstraintVals) in the set"
Base.length(conSet::ConstraintSet) = length(conSet.constraints)
Base.copy(conSet::ConstraintSet) = ConstraintSet(conSet.n, conSet.m,
	copy(conSet.constraints), copy(conSet.p), copy(conSet.c_max))


"Re-calculate the number of constraints in the constraint set"
function num_constraints!(conSet::ConstraintSet)
	p = conSet.p
	p .*= 0
	for con in conSet.constraints
		for k = 1:length(p)
			p[k] += length(con, k)
		end
	end
end

"""```julia
num_constraints(::ConstraintSet)
num_constraints(::AbstractSolver)
num_constraints(::Problem)
```
Get the total number of constraints at each time step
"""
@inline num_constraints(conSet::ConstraintSet) = conSet.p

"""```julia
add_constraint!(conSet, conVal::ConstraintVals, idx=-1)
add_constraint!(conSet, con::AbstractConstraint, inds::UnitRange, idx=-1)
```
Add a constraint to the constraint set. You can directly add a `ConstraintVals` type,
but when adding a normal `AbstractConstraint` you must specify the range of knotpoints
to which the constraint applies.

The optional `idx` argument allows you to specify exactly where in the vector of constraints
you'd like the constraint to be added. The ordering will effect the order in which the constraint
appears in concatenated vectors of constraint values (that show up in direct methods), potentially
effecting the band structure of the resultant Jacobian. See solvers for more details.
"""
function add_constraint!(conSet::ConstraintSet, conVal::ConstraintVals, idx=-1)
	if idx == -1
		push!(conSet.constraints, conVal)
		push!(conSet.c_max, 0)
	else
		insert!(conSet.constraints, idx, conVal)
		insert!(conSet.c_max, idx, 0)
	end
	num_constraints!(conSet)
end

function add_constraint!(conSet::ConstraintSet, con::AbstractConstraint,
		inds::UnitRange, idx=-1)
	conVal = ConstraintVals(con, inds)
	add_constraint!(conSet, conVal, idx)
end


"""```julia
max_violation(conSet::ConstraintSet)
max_violation(conSet::ConstraintSet, Z::Traj)
max_violation(prob::Problem, Z=prob.Z)
max_violation(solver::AbstractSolver)
max_violation(solver::AbstractSolver, Z)
```
Calculate the maximum constraint violation for the entire constraint set.
	If the a trajectory is not passed in, the violation is computed from the currently
	stored constraint values; otherwise, the constraints are re-computed using the
	trajectory passed in.
"""
function max_violation(conSet::ConstraintSet)
	max_violation!(conSet)
	maximum(conSet.c_max)
end

function max_violation(conSet::ConstraintSet, Z::Traj)
	evaluate!(conSet, Z)
	max_violation(conSet)
end

function max_violation!(conSet::ConstraintSet{T}) where T
	for i in eachindex(conSet.constraints)
		con = conSet.constraints[i]
		max_violation!(con)
		conSet.c_max[i] = maximum(con.c_max::Vector{T})
	end
end

"Calculate the maximum penalty parameter across all constraints"
function max_penalty(conSet::ConstraintSet)
	max_penalty!(conSet)
	maximum(conSet.c_max)
end

function max_penalty!(conSet::ConstraintSet{T}) where T
	for (i,con) in enumerate(conSet.constraints)
		max_penalty!(con)
		conSet.c_max[i] = maximum(con.c_max::Vector{T})
	end
end


"""```julia
evaluate!(conSet::ConstraintSet, Z::Traj)
```
Compute constraint values for all constraints for the entire trajectory
"""
function evaluate!(conSet::ConstraintSet, Z::Traj)
	for con in conSet.constraints
		evaluate!(con, Z)
	end
end

"""```julia
jacobian!(conSet::ConstraintSet, Z::Traj)
```
Compute constraint Jacobians for all constraints for the entire trajectory
"""
function jacobian!(conSet::ConstraintSet, Z::Traj)
	for con in conSet.constraints
		jacobian!(con, Z)
	end
end

"""```julia
update_active_set!(conSet::ConstraintSet, Z::Traj, ::Val{tol})
```
Compute the active set for the current constraint values, with tolerance tol.
	Uses a value type to avoid an allocation down the line.
"""
function update_active_set!(conSet::ConstraintSet, Z::Traj, val::Val{tol}=Val(0.0)) where tol
	for con in conSet.constraints
		update_active_set!(con, val)
	end
end

Base.iterate(conSet::ConstraintSet) = @inbounds (conSet.constraints[1], 1)
Base.iterate(conSet::ConstraintSet, i) = @inbounds i >= length(conSet.constraints) ? nothing : (conSet.constraints[i+1], i+1)
@inline Base.getindex(conSet::ConstraintSet, i) = conSet.constraints[i]

"Reset all the Lagrange multipliers and constraint values to zero and
	penalties the their initial value"
function reset!(conSet::ConstraintSet)
	for con in conSet.constraints
		reset!(con)
	end
end

function change_dimension(conSet::ConstraintSet, n, m)
	cons = map(conSet.constraints) do con

		# Check if the new dimensions match the old ones
		if check_dims(con,n,m)
			return con
		end

		con_idx = IndexedConstraint(n,m,con.con)
		con_val = ConstraintVals(con_idx, con.inds)
	end
	ConstraintSet(n,m,cons, length(conSet.p))
end
