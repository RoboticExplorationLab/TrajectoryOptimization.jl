```@meta
CurrentModule = TrajectoryOptimization
```

# 1. Setting Up a Dynamics Model
TrajectoryOptimization relies on the interface defined by [RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl) to define the forced dynamics required to solve
the problem. Please refer to the documentation for more details on setting up and defining
models. We present a simple example here.

Assume we want to find an optimal trajectory for the canonical cartpole system. We can either
import the existing model defined in [RobotZoo.jl](https://github.com/bjack205/RobotZoo.jl) or
use RobotDynamics.jl to define our own. Defining our own model is pretty straight-forward:

```julia
struct Cartpole{T} <: AbstractModel
    mc::T
    mp::T
    l::T
    g::T
end

Cartpole() = Cartpole(1.0, 0.2, 0.5, 9.81)

function dynamics(model::Cartpole, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q  = x[ SA[1,2] ]  # SA[...] creates a StaticArray.
    qd = x[ SA[3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end

RobotDynamics.state_dim(::Cartpole) = 4
RobotDynamics.control_dim(::Cartpole) = 1
```

with our dynamics model defined, we are ready to start setting up the optimization problem.

!!! tip
    For best performance, use [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), which offers loop-unrolling and allocation-free methods for 
    small to medium-sized matrices and vectors.

