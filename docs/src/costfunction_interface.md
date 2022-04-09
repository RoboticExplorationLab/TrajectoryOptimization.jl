```@meta
CurrentModule = TrajectoryOptimization
```

All cost functions in TrajectoryOptimization.jl inherit from 
`RobotDynamics.ScalarFunction`, and leverage that interface. This allows 
the `RobotDynamics.@autodiff` method to be used to automatically generate
efficient methods to evaluate the gradient and Hessian. 

We give an example of defining a new user-defined cost function in the 
example below, to illustrate how the interface works.

# Cost Function Interface
Here we define a nonlinear cost function for the cartpole system:

```math
    Q_2 * cos(\theta / 2) + \frac{1}{2} (Q_1 y^2 + Q_3 \dot{y}^2 + Q_4 \dot{\theta}^2) + \frac{1}{2} R ^2
```

We start by defining a new struct that inherits from 
`TrajectoryOptimization.CostFunction`:

```@example
using TrajectoryOptimization
using RobotDynamics
using ForwardDiff
using FiniteDiff

RobotDynamics.@autodiff struct CartpoleCost{T} <: TrajectoryOptimization.CostFunction
    Q::Vector{T}
    R::Vector{T}
end

RobotDynamics.state_dim(::CartpoleCost) = 4
RobotDynamics.control_dim(::CartpoleCost) = 1

function RobotDynamics.evaluate(cost::CartpoleCost, x, u)
    y = x[1]
    θ = x[2]
    ydot = x[3]
    θdot = x[4]
    J = cost.Q[2] * cos(θ/2)
    J += 0.5* (cost.Q[1] * y^2 + cost.Q[3] * ydot^2 + cost.Q[4] * θdot^2)
    if !isempty(u)
        J += 0.5 * cost.R[1] * u[1]^2 
    end
    return J
end
```

!!! note 
    Note that we check to see if `u` was empty, which can be the case at the
    last time step, depending on how a solver handles this case. It's usually
    a good idea to add a check like this.

!!! tip
    The `RobotDynamics.@autodiff` macro automatically defines the `gradient!`
    and `hessian!` methods from RobotDynamics.jl for us, using ForwardDiff.jl
    and FiniteDiff.jl.



All cost functions are required to define the following methods
```julia
n = RobotDynamics.state_dim(cost)
m = RobotDynamics.control_dim(cost)
RD.evaluate(cost, x, u)
```
and inherit from `CostFunction`. Note the it is good practice to use the method defined on
the terminal state internal to the method defined for both the state and control, i.e.
`gradient!(E, cost, x, u)` should call `gradient!(E, cost, xN)`.
They then inherit the following methods defined on knot points:

```julia
stage_cost(::CostFunction, ::KnotPoint)
gradient!(::QuadraticCostFunction, ::CostFunction, ::AbstractKnotPoint)
hessian!(::QuadraticCostFunction, ::CostFunction, ::AbstractKnotPoint)
```


# Objective Interface
The objective interface is very simple. After inheriting from `AbstractObjective`, define
the following methods:
```julia
Base.length(::NewObjective)       # number of knot points
get_J(::NewObjective)             # return vector of costs at each knot point
cost!(::NewObjective, Z::Traj)    # calculate the cost at each knot point and store in get_J(::NewSolver)
cost_expansion!(E::CostExpansion, obj::NewObjective, Z::Traj)
```

And inherits the single method
```julia
cost(::NewObjective, Z::Traj)
```
that simply returns the summed cost.
