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
gradient!(E::QuadraticCostFunction, cost, x, u)
gradient!(E::QuadraticCostFunction, cost, xN)
hessian!(E::QuadraticCostFunction, cost, x, u)
hessian!(E::QuadraticCostFunction, cost, xN)
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
