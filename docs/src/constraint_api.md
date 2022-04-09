```@meta
CurrentModule = TrajectoryOptimization
```

# [Constraints](@id constraint_api)
This page provides details about the various types in TrajectoryOptimization.jl for working
with constraints, as well as the methods defined on those types.

## Constraint List
A [`ConstraintList`](@ref) is used to define a trajectory optimization [`Problem`](@ref)
and only holds basic information about the constraints included in the problem. 
```@docs
ConstraintList
add_constraint!
num_constraints
constraintindices
functionsignature
diffmethod
gen_jacobian
```

## Implemented Constraints
The following is a list of the constraints currently implemented in TrajectoryOptimization.jl.
Please refer to the docstrings for the individual constraints on details on their constructors,
since each constraint is unique, in general.

List of currently implemented constraints
* [`GoalConstraint`](@ref)
* [`BoundConstraint`](@ref)
* [`LinearConstraint`](@ref)
* [`CircleConstraint`](@ref)
* [`CollisionConstraint`](@ref)
* [`SphereConstraint`](@ref)
* [`NormConstraint`](@ref)
* [`IndexedConstraint`](@ref)

```@docs
GoalConstraint
BoundConstraint
LinearConstraint
CircleConstraint
SphereConstraint
CollisionConstraint
NormConstraint
IndexedConstraint
```
