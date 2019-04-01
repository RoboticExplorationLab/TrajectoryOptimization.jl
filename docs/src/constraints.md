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
TerminalConstraint
ConstraintType
Equality
Inequality
```
There are two constraint types that inherit from AbstractConstraint: [Constraint](@ref) and [TerminalConstraint](@ref). Both of these constraints are parameterized by a [ConstraintType](@ref), which can be either [Equality](@ref) or [Inequality](@ref). This allows the software to easily dispatch over the type of constraint. Each constraint type represents a vector-valued constraint. The intention is that each constraint type represent one line in constraints of problem definition (where they may be vectoxr or scalar-valued). Each constraint contains the following fields:
* `c`: the in-place constraint function. Of the form `c(v,x,u)` for [Constraint](@ref) and `c(v,x)` for [TerminalConstraint](@ref).
* `∇c`: the in-place constraint jacobian function. For [Constraint](@ref) it can either be called as `∇c(A,B,x,u)` where `A` is the state Jacobian and `B` is the control Jacobian, or as `∇c(Z,x,u)` where `Z` is the p × (n+m) concatenated Jacobian. For [TerminalConstraint](@ref) there is only `∇c(A,x)`.
* `p`: number of elements in the constraint vector
* `label`: a Symbol for identifying the constraint

# Creating Constraints
A stage-wise constraint can be created with either of the two constructors
```
Constraint{S}(c::Function,∇c::Function,p::Int,label::Symbol) where S<:ConstraintType
Constraint{S}(c::Function,n::Int,m::Int,p::Int,label::Symbol) where S<:ConstraintType
```
The first is the default constructor. `c` must be in-place of the form `c(v,x,u)` where `v` holds the constraint function values. `∇c` must be multiple dispatched to have the forms `∇c(A,B,x,u)` where `A` is the state Jacobian and `B` is the control Jacobian, and `∇c(Z,x,u)` where `Z` is the p × (n+m) concatenated Jacobian.

The second will use ForwardDiff to generate the constraint Jacobian, so requires the size of the state and control input vectors.

A terminal constraint can be similarly defined using one of the following constructors

```
TerminalConstraint{S}(c::Function,∇c::Function,p::Int,label::Symbol) where S<:ConstraintType
TerminalConstraint{S}(c::Function,n::Int,p::Int,label::Symbol) where S<:ConstraintType
Constraint{S}(c::Function,n::Int,p::Int,label::Symbol) where S<:ConstraintType
```
which are identical to the ones above, expect that they require a constraint function and Jacobian of the form `c(v,x)` and `∇c(A,x)`.

## Special Constraints
A few constructors for common constraints have been provided:

```@docs
bound_constraint
```
