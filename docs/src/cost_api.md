```@meta
CurrentModule = TrajectoryOptimization
```

# Cost Functions and Objectives
This page details the functions related to building and evaluating cost functions and objectives.

## Quadratic Cost Functions
```@docs
QuadraticCost
LQRCost
LQRCostTerminal
LQRObjective
```

## Indexed Cost Functions
```@docs
IndexedCost
```

## Adding Cost Functions
Right now, TrajectoryOptimization supports addition of `QuadraticCost`s, but extensions to
general cost function addition should be straightforward, as long as the cost function all
have the same state and control dimensions.

Adding quadratic cost functions:
```julia
n,m = 4,5
Q1 = Diagonal(@SVector [1.0, 1.0, 1.0, 1.0, 0.0])
R1 = Diagonal(@SVector [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Q2 = Diagonal(@SVector [1.0, 1.0, 1.0, 1.0, 2.0])
R2 = Diagonal(@SVector [0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
cost1 = QuadraticCost(Q1, R1)
cost2 = QuadraticCost(Q2, R2)
cost3 = cost1 + cost2
# cost3 is equivalent to QuadraticCost(Q1+Q2, R1+R2)
```

## CostExpansion Type
The `CostExpansion` type stores the pieces of the second order Taylor expansion of the cost for the entire trajectory, stored as vectors of Static Vectors or Static Matrices. e.g. to get the Hessian with respect to `x` at knotpoint 5 you would use `E.xx[5]`.
```@docs
CostExpansion
```

## Objective
```@docs
Objective
cost(::Objective, Z::Traj)
get_J
cost_gradient
cost_gradient!
cost_hessian
cost_hessian!
```
