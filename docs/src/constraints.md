# [3. Creating Constraints](@id constraint_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["constraints.md"]
```

## Creating Constraint Sets
The easiest way to set up the constraints for your problem is through the [`ConstraintList`](@ref).
This structure simply holds a vector of all the constraints in the trajectory optimization problem.
The easiest way to start is to create an empty [`ConstraintList`](@ref):
```julia
cons = ConstraintList(n,m,N)
```

### Adding Constraints
You can add any constraint (the list of currently implemented constraints is given in the following
section) to the constraint set using the [`add_constraint!`](@ref) method. For example, if we
want to add control limits and an final goal constraint to our problem, we do this by creating
a `ConstraintList` and subsequently adding the constraints:
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
bnd = BoundConstraint(n,m, u_min=-ubnd, u_max=ubnd)
add_constraint!(cons, bnd, 1:N-1)  # add to all but the last time step
```

### Defined Constraints
The following constraints are currently defined. See their individual docstrings on details
on how to construct them, since constraint constructors are, in general, unique to the constraint.

* [`GoalConstraint`](@ref)
* [`BoundConstraint`](@ref)
* [`CircleConstraint`](@ref)
* [`SphereConstraint`](@ref)
* [`NormConstraint`](@ref)
* [`IndexedConstraint`](@ref)


### Working with `ConstraintList`
The [`ConstraintList`](@ref) type is useful type that provides some
convenient methods for manipulating and working with the constraints in your
trajectory optimization problem.

It supports many of the operations that work on an `AbstractVector`, such as
`length`, indexing (e.g. `cons[1]` or `cons[end]`), and iteration (e.g. `[con
for con in cons]`).

The time step indices can be retrieved via the `inds` field, e.g. `cons.inds[1]` returns the indices for the first constraint. You can also iterate over constraints and their time indices simultaneously using `zip(::ConstraintList)`, e.g. `[(length(con), length(inds)) for (con,inds) in zip(cons)]`. 

The `num_constraints(::ConstraintList)` returns a vector of length `N` with
total number of constraint values per time step. Note that since each
constraint is vector-valued, this is different than
`length(::ConstraintList)`. The total number of dual variables can be
calculated using `sum(num_constraints(cons))`.

The constraint list can also be sorted via `sort!(::ConstraintList)` to
partition it into constraints that apply each time step (`StageConstraint`)
and those that apply across two time steps (`CoupledConstraint`), such as
dynamics constraints.

To query if a constraint list contains a dynamics constraint, use `has_dynamics_constraint(::ConstraintList)`.