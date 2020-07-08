# TrajectoryOptimization

![Build Status](https://travis-ci.org/RoboticExplorationLab/TrajectoryOptimization.jl.svg?branch=master)
![CI](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/dev)

A library of solvers for trajectory optimization problems written in Julia. Currently, the following methods are implemented with a common interface:

[ALTRO (Augmented Lagrangian TRajectory Optimizer)](https://rexlab.stanford.edu/papers/altro-iros.pdf): A fast solver for constrained trajectory optimization problems formulated as MDPs that features:
  * General nonlinear cost functions, including minimum time problems
  * General nonlinear state and input constraints
  * Infeasible state initialization
  * Square-root methods for improved numerical conditioning
  * Active-set projection method for solution polishing

Direct Collocation (DIRCOL)
  * Interfaces to Nonlinear Programming solvers (e.g., [Ipopt](https://github.com/coin-or/Ipopt), [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/)) via [MathOptInterface](https://github.com/JuliaOpt/MathOptInterface.jl)

All methods utilize Julia's extensive autodifferentiation capabilities via [ForwardDiff.jl](http://www.juliadiff.org/ForwardDiff.jl/) so that the user does not need to specify derivatives of dynamics, cost, or constraint functions.

## Installation
To install TrajectoryOptimization.jl, run the following from the Julia REPL:
```julia
Pkg.add("TrajectoryOptimization")
```

# What's New
`TrajectoryOptimization.jl` underwent significant changes between versions `v0.1` and `v0.2`. The new code is significantly faster (up to 100x faster). The core part of the ALTRO solver (everything except the projected newton phase) is completely allocation-free once the solver has been initialized. Most of the API has changed significantly. See the documentation for more information on the new API.

## Quick Start
To run a simple example of a constrained 1D block move (see script in `/examples/quickstart.jl`):
```julia
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
const TO = TrajectoryOptimization

struct DoubleIntegrator{T} <: AbstractModel
    mass::T
end

function TO.dynamics(model::DoubleIntegrator, x, u)
    SA[x[2], u[1] / model.mass]
end

Base.size(::DoubleIntegrator) = 2,1

# Model and discretization
model = DoubleIntegrator(1.0)
n,m = size(model)
tf = 3.0  # sec
N = 21    # number of knot points

# Objective
x0 = SA[0,0.]  # initial state
xf = SA[1,0.]  # final state

Q = Diagonal(@SVector ones(n))
R = Diagonal(@SVector ones(m))
obj = LQRObjective(Q, R, N*Q, xf, N)

# Constraints
cons = TO.ConstraintSet(n,m,N)
add_constraint!(cons, GoalConstraint(xf), N:N)
add_constraint!(cons, BoundConstraint(n,m, u_min=-10, u_max=10), 1:N-1)

# Create and solve problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
solver = ALTROSolver(prob)
cost(solver)           # initial cost
solve!(solver)         # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# Get the state and control trajectories
X = states(solver)
U = controls(solver)
```

## Examples
Notebooks with more detailed examples can be found [here](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl/tree/master/examples)

## Related Papers
* [IROS 2019 paper](https://rexlab.stanford.edu/papers/altro-iros.pdf).
