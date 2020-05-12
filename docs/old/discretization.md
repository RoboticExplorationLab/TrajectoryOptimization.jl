```@meta
CurrentModule = TrajectoryOptimization
```

# Discretization
This page gives details on the methods for evaluating discretized dynamics, as well as instructions
on how to define a custom integration method.

## Model Discretization
With a model defined, we can compute the discrete dynamics and discrete dynamics Jacobians for an Implicit
integration rule with the following methods

```@docs
discrete_dynamics
discrete_jacobian
```

## Integration Schemes
TrajectoryOptimization.jl has already defined a handful of integration schemes for computing discrete dynamics.
The integration schemes are specified as abstract types, so that methods can efficiently dispatch
based on the integration scheme selected. Here is the current set of implemented types:
* [`QuadratureRule`](@ref)
    * [`Implicit`](@ref)
        * [`RK3`](@ref)
    * [`Explicit`](@ref)
        * [`HermiteSimpson`](@ref)

```@docs
QuadratureRule
Implicit
RK3
Explicit
HermiteSimpson
```

## Defining a New Integration Scheme

### Implicit Methods
Implicit integration schemes are understandably simpler, since the output is not a function of
itself, as is the case with explicit jschemes. As such, as a minimum, the user only needs to define
the following method for a new rule `MyQ`:

```julia
x′ = discrete_dynamics(::Type{MyQ}, model::AbstractModel, x, u, dt)
```

### Explicit Methods
Explicit integration schemes are specified with a [`DynamicsConstraint`](@ref). These methods
are most efficiently computed when the entire trajectory is considered at once, thereby avoiding
duplicate function evaluations. As a result, the user must define methods that deal with the
entire trajectory at once:

```julia
evaluate!(vals::Vector{<:AbstractVector}, con::DynamicsConstraint{MyQ},
    Z::Traj, inds=1:length(Z)-1)
```
Here `vals` is a Vector of Static Vectors, where the result of the calculation will be stored.
`con` is a `DynamicsConstraint` that specifies the integration scheme, `Z` is the trajectory,
and `inds` are the knotpoints where the constraint is applied (which should always be 1:N-1
if you have a single model for the entire trajectory). The method should compute
```julia
vals[k] = x[k+1] - f(x[k],u[k],x[k+1],u[k+1])
```
which is the amount of dynamic infeasibility between knotpoints. The method should obviously
loop over the entire trajectory (see implementation for `HermiteSimpson`).

### Integrating Cost Functions
Some methods, such as DIRCOL, apply the integration scheme to the cost function, as well.
This can be done for a new integration rule by defining the following methods:
```julia
cost(obj::Objective, dyn_con::DynamicsConstraint{MyQ}, Z::Traj)
cost_gradient!(E::CostExpansion, obj::Objective, dyn_con::DynamicsConstraint{MyQ}, Z::Traj)
```
