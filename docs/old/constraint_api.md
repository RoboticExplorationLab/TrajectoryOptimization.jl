```@meta
CurrentModule = TrajectoryOptimization
```

# Constraints
This page provides details about the various types in TrajectoryOptimization.jl for working
with constraints, as well as the methods defined on those types.


## Constraint Sets
```@docs
ConstraintSet
```

### Information Methods
These methods provide or calculate information about the constraint set
```@docs
Base.size(::ConstraintSet)
Base.length(::ConstraintSet)
num_constraints!
num_constraints
```

### Calculation Methods
These methods perform calculations on the constraint set
```@docs
max_violation
max_penalty
evaluate!(::ConstraintSet, ::Traj)
jacobian!(::ConstraintSet, ::Traj)
reset!(::ConstraintSet)
```
`ConstraintSet` supports indexing and iteration, which returns the `ConstraintVals` at that index. However, to avoid allocations, iteration directly on the `.constraints` field.

Additionally, to avoid allocations when computing `max_violation`, you can call `max_violation!(conSet)` and then `maximum(conSet.c_max)` to perform the reduction in the scope where the result is stored (thereby avoiding an allocation).

### Changing Dimension
```@docs
change_dimension(conSet::ConstraintSet, n, m)
```

## Implemented Constraints
The following is a list of the constraints currently implemented in TrajectoryOptimization.jl.
Please refer to the docstrings for the individual constraints on details on their constructors,
since each constraint is unique, in general.

List of currently implemented constraints
* [`GoalConstraint`](@ref)
* [`BoundConstraint`](@ref)
* [`CircleConstraint`](@ref)
* [`SphereConstraint`](@ref)
* [`NormConstraint`](@ref)
* [`DynamicsConstraint`](@ref)
* [`IndexedConstraint`](@ref)

```@docs
GoalConstraint
BoundConstraint
CircleConstraint
SphereConstraint
NormConstraint
DynamicsConstraint
IndexedConstraint
```

## `ConstraintVals` Type
```@docs
ConstraintVals
```
