```@meta
CurrentModule = TrajectoryOptimization
```

# Cost Function Interface
All cost functions are required to define the following methods
```julia
n = state_dim(cost)
m = control_dim(cost)
J = stage_cost(cost, x, u)
J = stage_cost(cost, xN)
Qx,Qu = gradient(cost, x, u)
Qxx,Quu,Qux = hessian(cost, x, u)
```
and inherit from `CostFunction`. They then inherit the following methods:

```@docs
stage_cost(::CostFunction, ::KnotPoint)
cost_gradient(::CostFunction, ::KnotPoint)
cost_hessian(::CostFunction, ::KnotPoint)
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
