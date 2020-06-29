```@meta
CurrentModule = TrajectoryOptimization
```

# Changing Problem Dimension
It is common practice to augment the states or controls with additional terms in order to
help solve the problem. This page describes experimental methods built into TrajectoryOptimization.jl
that make this process easier and hopefully natural.

For example, let's say we have trajectory optimization problem for the canonical cartpole,
similar to the one described in [Creating a Problem](@ref). Now, for some reason, we want to
augment the state and control dimension by 1, each, so that the new state and control dimensions
are 5 and 2, respectively. We'd like to keep the cost function and constraints we've already
defined, and simply add on a few

## Cost Function
Let's say we have model with state and control dimension of 4 and 1, respectively. Suppose
now we augment our state and control dimensions to 6 and 3, for some reason. We likely already
have a cost function defined for our original problem. We'd like to augment this cost function
with the costs on the new s
