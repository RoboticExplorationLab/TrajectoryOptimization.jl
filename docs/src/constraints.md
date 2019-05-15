# 4. Add Constraints
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["constraints.md"]
```

# Constraint Type
```@docs
AbstractConstraint
Constraint
ConstraintType
Equality
Inequality
```
There are two constraint types that inherit from AbstractConstraint: [Constraint](@ref) and [TerminalConstraint](@ref). Both of these constraints are parameterized by a [ConstraintType](@ref), which can be either [Equality](@ref) or [Inequality](@ref). This allows the software to easily dispatch over the type of constraint. Each constraint type represents a vector-valued constraint. The intention is that each constraint type represent one line in constraints of problem definition (where they may be vector or scalar-valued). Each constraint contains the following fields:
* `c`: the in-place constraint function. Methods dispatch over constraint functions of the form `c(v,x,u)` and `c(v,x)`.
* `∇c`: the in-place constraint jacobian function defined as `∇c(Z,x,u)` where `Z` is the p × (n+m) concatenated Jacobian. Methods also dispatch over constraint jacobians of the form `∇c(Z,x)` where `Z` is p x n.
* `p`: number of elements in the constraint vector
* `label`: a Symbol for identifying the constraint

# Creating Constraints
A stage-wise constraint can be created with either of the two constructors
```
Constraint{S}(c::Function,∇c::Function,p::Int,label::Symbol) where S<:ConstraintType
Constraint{S}(c::Function,n::Int,m::Int,p::Int,label::Symbol) where S<:ConstraintType
```
The first is the default constructor. `c` must be in-place of the form `c(v,x,u)` where `v` holds the constraint function values.

The second will use ForwardDiff to generate the constraint Jacobian, so requires the size of the state and control input vectors.

## Special Constraints
A few constructors for common constraints have been provided:

```@docs
BoundConstraint
goal_constraint
```
