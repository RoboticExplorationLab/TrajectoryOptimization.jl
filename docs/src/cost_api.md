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

## CostExpansion Type
The `CostExpansion` type stores the pieces of the second order Taylor expansion of the cost for the entire trajectory, stored as vectors of Static Vectors or Static Matrices. e.g. to get the Hessian with respect to `x` at knotpoint 5 you would use `E.xx[5]`.
```@docs
CostExpansion
```

## Objective
```@docs
Objective
cost
stage_cost
get_J
cost_gradient
cost_gradient!
cost_hessian
cost_hessian!
```
