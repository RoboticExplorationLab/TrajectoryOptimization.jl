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

#### Information Methods
These methods provide or calculate information about the constraint set
```@docs
Base.size(::ConstraintSet)
Base.length(::ConstraintSet)
num_constraints!
num_constraints
```

#### Calculation Methods
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

## Implemented Constraints
The following is a list of the constraints currently implemented in TrajectoryOptimization.jl.
Please refer to the docstrings for the individual constraints on details on their constructors,
since each constraint is unique, in general.

```@docs
GoalConstraint
BoundConstraint
CircleConstraint
SphereConstraint
NormConstraint
IndexedConstraint
```

## Constraint Type
All constraints inherit from `AbstractConstraint{S<:ConstraintSense,W<:ConstraintType,P}`,
where `ConstraintSense` specifies `Inequality` or `Equality`, `ConstraintType` specifies the
"bandedness" of the constraint (will be discussed more later), and `P` is the dimension of
the constraint. This allows the software to easily dispatch over the type of constraint.
Each constraint type represents a vector-valued constraint.
The intention is that each constraint type represent one line in the constraints of
problem definition (where they may be vector or scalar-valued).

TrajectoryOptimization.jl assumes equality constraints are of the form ``g(x) = 0`` and inequality
constraints are of the form ``h(x) \leq 0 ``.

```@docs
AbstractConstraint
ConstraintSense
ConstraintType
```

### Methods
The following methods are defined for all `AbstractConstraint`s
```@docs
state_dims
control_dims
evaluate!
jacobian!
contype
sense
width
upper_bound
lower_bound
is_bound
check_dims
```

## Adding a New Constraint
See interface description in documentation for [`AbstractConstraint`](@ref). The
interface allows for a lot of flexibility, but let's do a simple example. Let's say
we have a 2-norm constraint on the controls at each time step, e.g. ``||u|| \leq a``.
We can do this with just a few lines of code:

```julia
struct ControlNorm{T} <: AbstractConstraint{Inequality,Control,1}
  m::Int
  a::T
end
control_dim(con::ControlNorm) = con.m
evaluate(con::ControlNorm, u::SVector) = @SVector [norm(u) - con.a] # needs to be a vector output
jacobian(con::ControlNorm, u::SVector) = u'/norm(u)  # optional
```
Importantly, note that the inheritance specifies the constraint applies only to
individual controls, the constraint in an inequality, and has dimension 1.

Let's say the bound ``a`` varied by time-step. We could handle this easily by instead defining the methods operating on the entire trajectory:

```julia
struct ControlNorm2{T} <: AbstractConstraint{Inequality,Control,1}
  m::Int
  a::Vector{T}
end
control_dim(con::ControlNorm) = con.m
function evaluate!(vals::Vector{<:AbstractVector}, con::ControlNorm,
    Z, inds=1:length(Z)-1)
  for (i,k) in enumerate(inds)
    u = control(Z[k])
    vals[i] = @SVector [norm(u) - con.a[k]]
  end
end
function jacobian!(∇c::Vector{<:AbstractMatrix}, con::ControlNorm,
    Z, inds=1:length(Z)-1)
  for (i,k) in enumerate(inds)
    u = control(Z[k])
    ∇c[i] = u'/norm(u)
  end
end
```

## API
```@docs
ConstraintVals
Stage
State
Control
Coupled
Dynamical
CoupledState
CoupledControl
General
GeneralState
GeneralControl
```
