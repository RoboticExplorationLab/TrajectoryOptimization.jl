```@meta
CurrentModule = TrajectoryOptimization
```

# Constraints
This page provides details about the various types in TrajectoryOptimization.jl for working
with constraints, as well as the methods defined on those types.
In general, a [`ConstraintList`](@ref) is used to define the constraints, and another
[`AbstractConstraintSet`](@ref), such as an [`ALConstraintSet`](@ref), is instantiated by a
solver to hold the constraint values and Jacobians.

## Constraint List
A [`ConstraintList`](@ref) is used to define a trajectory optimization [`Problem`](@ref) and
only holds basic information about the constraints included in the problem. Although it is
a child of [`AbstractConstraintSet`](@ref) and supports indexing and iteration, it does not
hold any information about constraint values or Jacobians.
```@docs
ConstraintList
add_constraint!
num_constraints
```

## Constraint Sets
A constraint set holding a list of [`ConVal`](@ref)s is generally instantiated by a solver
and holds the constraint definitions, as well as the associated constraint values, Jacobians,
and other constraint-related information required by the solver.
```@docs
AbstractConstraintSet
ALConstraintSet
link_constraints!
```

## Constraint Value type
The [`ConVal`](@ref) type holds all the constraint values and Jacobians for a particular
constraint, and supports different ways of storing those (either as individual matrices/vectors
or as views into a large matrix/vector).

```@docs
ConVal
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
