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
NegativeOrthant
SecondOrderCone
```

## Evaluating Constraints
The following methods are used to evaluate a constraint:
```@docs
evaluate
evaluate!
jacobian!
∇jacobian!
```

### Methods
The following methods are defined for all `AbstractConstraint`s
```@docs
state_dim
control_dim
sense
widths
upper_bound
lower_bound
is_bound
check_dims
get_inds
```

## Adding a New Constraint
See interface description in documentation for [`AbstractConstraint`](@ref). The
interface allows for a lot of flexibility, but let's do a simple example. Let's say
we have a 2-norm constraint on the controls at each time step, e.g. ``||u|| \leq a``.
We can do this with just a few lines of code:

```julia
struct ControlNorm{T} <: ControlConstraint
	m::Int
	val::T
	function ControlNorm(m::Int, val::T) where T
		@assert val ≥ 0 "Value must be greater than or equal to zero"
		new{T}(m,val,sense,inds)
	end
end
control_dim(con::ControlNorm) = con.m
sense(::ControlNorm) = Inequality()
Base.length(::ControlNorm) = 1
evaluate(con::ControlNorm, u::SVector) = SA[norm(u) - con.a] # needs to be a vector output
jacobian(con::ControlNorm, u::SVector) = u'/norm(u)  # optional
```
Importantly, note that the inheritance specifies the constraint applies only to
individual controls.

Let's say the bound ``a`` varied by time-step. We could handle this easily by instead defining the methods operating on the entire trajectory:

```julia
struct ControlNorm2{T} <: ControlConstraint
	m::Int
	val::Vector{T}
	function ControlNorm2(m::Int, val::T) where T
		@assert val ≥ 0 "Value must be greater than or equal to zero"
		new{T}(m,val,sense,inds)
	end
end
control_dim(con::ControlNorm2) = con.m
sense(::ControlNorm2) = Inequality()
Base.length(::ControlNorm2) = 1
function evaluate!(vals, con::ControlNorm2, Z::AbstractTrajectory, inds=1:length(Z))
	for (i,k) in enumerate(inds)
		u = control(Z[k])
		vals[i] = SA[norm(u) - con.a[i]]
	end
end
function jacobian!(∇c, con::ControlNorm2, Z::AbstractTrajectory, inds=1:length(Z),
		is_const = BitArray(undef, size(∇c)))
	for (i,k) in enumerate(inds)
			u = control(Z[k])
			∇c[i] = u'/norm(u)
			is_const[i] = false
	end
end
```

### Constraint Types
The `ConstraintType` defines the "bandedness" of the constraint, or the number of adjacent
state or constraint values needed to calculate the constraint.
```@docs
StageConstraint
StateConstraint
ControlConstraint
CoupledConstraint
CoupledStateConstraint
CoupledControlConstraint
```
