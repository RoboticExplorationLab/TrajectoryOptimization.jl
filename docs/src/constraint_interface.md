```@meta
CurrentModule = TrajectoryOptimization
```

# Constraint Interface
All constraints inherit from [`AbstractConstraint`](@ref).
```@docs
AbstractConstraint
```

## Constraint Sense 
TrajectoryOptimization.jl assumes equality constraints are of the form ``g(x) = 0``, 
and that all other constraints are constrained to lie with a specified cone. This 
is referred to as the `ConstraintSense`. The following are currently implemented:

```@docs
ConstraintSense
Equality
Inequality
ZeroCone
NegativeOrthant
SecondOrderCone
```

## Evaluating Constraints
The following methods are used to evaluate a constraint:
```@docs
evaluate_constraints!
constraint_jacobians!
```

### Methods
The following methods are defined for all `AbstractConstraint`s
```@docs
sense
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
using TrajectoryOptimization
using RobotDynamics
using ForwardDiff
using FiniteDiff

RobotDynamics.@autodiff struct ControlNorm{T} <: TrajectoryOptimization.ControlConstraint
    m::Int
    val::T
    function ControlNorm(m::Int, val::T) where T
        @assert val ≥ 0 "Value must be greater than or equal to zero"
        new{T}(m,val,sense,inds)
    end
end
RobotDynamics.control_dim(con::ControlNorm) = con.m
RobotDynamics.output_dim(::ControlNorm) = 1
TrajectoryOptimization.sense(::ControlNorm) = Inequality()

RobotDynamics.evaluate(con::ControlNorm, u) = SA[norm(u) - con.a] # needs to be a vector output
RobotDynamics.evaluate!(con::ControlNorm, c, u) = SA[norm(u) - con.a]

function RobotDynamics.jacobian!(con::ControlNorm, J, c, u)  # optional
    J[1,:] .= u'/norm(u)
end
```
Importantly, note that the inheritance specifies the constraint applies only to
individual controls.

### Constraint Types
The `ConstraintType` defines the whether the constraint is a function of just the 
state, control, or both the state and control. This automatically defines the 
`RobotDynamics.FunctionInputs` trait for the constraint.

```@docs
StageConstraint
StateConstraint
ControlConstraint
```
