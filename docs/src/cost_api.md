```@meta
CurrentModule = TrajectoryOptimization
```

# Cost Functions and Objectives
This page details the functions related to building and evaluating cost functions and objectives.


## Cost Functions
```@docs
CostFunction
QuadraticCostFunction
DiagonalCost
QuadraticCost
LQRCost
is_diag
is_blockdiag
invert!
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

## Objectives
```@docs
Objective
LQRObjective
get_J
dgrad
dhess
norm_grad
```

## Evaluating the Cost
```@docs
cost
stage_cost
gradient!
hessian!
cost_gradient!
cost_hessian!
cost_expansion!
```
