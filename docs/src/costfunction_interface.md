
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
and inherit from `CostFunction`.
