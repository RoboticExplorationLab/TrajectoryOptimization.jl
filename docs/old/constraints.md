# [3. Creating Constraints](@id constraint_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["constraints.md"]
```
## Creating Constraint Sets
The easiest way to set up the constraints for your problem is through the [`ConstraintSet`](@ref). This structure simply holds a vector of all the constraints in the trajectory optimization problem. The easiest way to start is to create an empty `ConstraintSet`:
```julia
conSet = ConstraintSet(n,m,N)
```

### Adding Constraints
You can add any constraint (the list of currently implemented constraints is given in the following section) to the constraint set using the following method:
```@docs
add_constraint!
```

### Defined Constraints
The following constraints are currently defined. See thier individual docstrings on details
on how to construct them, since constraint constructors are, in general, unique to the constraint.

* [`GoalConstraint`](@ref)
* [`BoundConstraint`](@ref)
* [`CircleConstraint`](@ref)
* [`SphereConstraint`](@ref)
* [`NormConstraint`](@ref)
* [`IndexedConstraint`](@ref)
