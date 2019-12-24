# [1. Setting up a Dynamics Model](@id model_section)
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["models.md"]
```
## Overview
The Model type holds information about the dynamics of the system. All dynamics are assumed to be state-space models of the system of the form ``\dot{x} = f(x,u)`` where ``\dot{x}`` is the state derivative, ``x`` an ``n``-dimensional state vector, and ``u`` in an ``m``-dimensional control input vector. The function ``f`` can be any nonlinear function.

TrajectoryOptimization.jl solves the trajectory optimization problem by discretizing the state and control trajectories, which requires discretizing the dynamics, i.e., turning the continuous time differential equation into a discrete time difference equation of the form ``x[k+1] = f(x[k],u[k])``, where ``k`` is the time step. There many methods of performing this discretization, and TrajectoryOptimization.jl offers several of the most common methods. See [Model Discretization](@ref) section for more information on
discretizing dynamics, as well as how to define custom integration methods.


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
import TrajectoryOptimization: dynamics  # the dynamics function must be imported

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

Base.size(::Cartpole) = 4,1
```

And voila! we have a new model.

We now have a few methods automatically available to us:
```@docs
dynamics
jacobian
```

### Time-varying systems (experimental)
TrajectoryOptimization.jl also offers experimental support for time-varying systems. Let's say
for some reason the mass of our cartpole is decreasing linearly with time. We can model this
with a slight modification to the dynamics function signature:

```julia
import TrajectoryOptimization: dynamics

struct CartpoleTimeVarying{T} <: AbstractModel
    mc::T  # initial mass of the cart
    mp::T  # mass of the pole
    l::T   # length of the pole
    g::T   # gravity
end

function dynamics(model::CartpoleTimeVarying, x, u, t)  # note extra time parameter
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    # Change the mass of the cart with time
    mc = mc - 0.01*t

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

Base.size(::CartpoleTimeVarying) = 4,1
```
