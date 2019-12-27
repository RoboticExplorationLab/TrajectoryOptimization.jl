```@meta
CurrentModule = TrajectoryOptimization
```

# Constraint Interface

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

### Constraint Types
The `ConstraintType` defines the "bandedness" of the constraint, or the number of adjacent
state or constraint values needed to calculate the constraint. 
```@docs
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
