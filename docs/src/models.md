# [1. Setting up a Dynamics Model](@id model_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["models.md"]
```
## Overview
The Model type holds information about the dynamics of the system. All dynamics are assumed to be state-space models of the system of the form ẋ = f(x,u) where ẋ is the state derivative, x an n-dimensional state vector, and u in an m-dimensional control input vector. The function f can be any nonlinear function.

TrajectoryOptimization.jl solves the trajectory optimization problem by discretizing the state and control trajectories, which requires discretizing the dynamics, i.e., turning the continuous time differential equation into a discrete time difference equation of the form x[k+1] = f(x[k],u[k]), where k is the time step. There many methods of performing this discretization, and TrajectoryOptimization.jl offers several of the most common methods.

## Creating a New Model
To create a new model of a dynamical system, you need to define a new type that inherits from `AbstractModel`. You will need to then define only a few methods on your type. Let's say we want to create a model of the canonical cartpole. We start by defining our type:
```julia
struct Cartpole{T} <: AbstractModel
    mc::T  # mass of the cart
    mp::T  # mass of the pole
    l::T   # length of the pole
    g::T   # gravity
end
```
It's often convenient to store any model parameters inside the new type (make sure they're concrete types!). If you need to store vectors or matrices, we highly recommend using StaticArrays, which are extremely fast and avoid memory allocations. For models with lots of parameters, we recommend [Parameters.jl](https://github.com/mauro3/Parameters.jl) that makes it easy to specify default parameters.

We now just need to define two functions to complete the interface
```julia
function dynamics(model::Cartpole, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end

size(::Cartpole) = 4,1
```

And voila! we have a new model.

We now have a few methods automatically available to us:
```@docs
dynamics
jacobian
```

The next section outlines how our continuous time model is discretized

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

### Defining a New Integration Scheme

#### Implicit Methods
Implicit integration schemes are understandably simpler, since the output is not a function of
itself, as is the case with implicit schemes. As such, as a minimum, the user only needs to define
the following method for a new rule `MyQ`:

```julia
x′ = discrete_dynamics(::Type{MyQ}, model::AbstractModel, x, u, dt)
```

#### Explicit Methods
Explicit integration schemes are specified in a [`DynamicsConstraint`](@ref). These methods
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

#### Integrating Cost Functions
Some methods, such as DIRCOL, apply the integration scheme to the cost function, as well.
This can be done for a new integration rule by defining the following methods:
```julia
cost(obj::Objective, dyn_con::DynamicsConstraint{MyQ}, Z::Traj)
cost_gradient!(E::CostExpansion, obj::Objective, dyn_con::DynamicsConstraint{MyQ}, Z::Traj)
```
