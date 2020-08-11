# TrajectoryOptimization

| *Build Status* | *Documentation* |
| -------------- | --------------- |
| ![Build Status](https://travis-ci.org/RoboticExplorationLab/TrajectoryOptimization.jl.svg?branch=master) ![CI](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl/workflows/CI/badge.svg) [![codecov](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl) | [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/dev) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/stable) |

This package is built for the express purpose of defining and evaluating trajectory optimization problems. Although early versions (pre v0.3) also included methods to solve these problems, this is now left to separate packages that implement the interface defined in TrajectoryOptimization.jl. For example, [Altro.jl](https://github.com/RoboticExplorationLab/Altro.jl) implements the ALTRO solver that used to be included in TrajectoryOptimization.jl. This change was done to make this package lighter and allow more abstraction in how solvers set up and solve the problems defined by this package.

TrajectoryOptimization.jl aims to provide both a convenient API for setting up and defining trajectory optimization problem and extremely efficient methods for evaluating them. Nearly all of the methods implemented have zero memory allocations and have been highly optimized for speed. Since trajectory optimization problem have a unique structure that set them apart from generic NLPs (nonlinear programs), use of the specialized methods in TrajectoryOptimization.jl can provide dramatic improvements in the computational efficiency of the solvers that implement the API.

All methods utilize Julia's extensive autodifferentiation capabilities via [ForwardDiff.jl](http://www.juliadiff.org/ForwardDiff.jl/) so that the user does not need to specify derivatives of dynamics, cost, or constraint functions.

## Installation
To install TrajectoryOptimization.jl, run the following from the Julia REPL:
```julia
Pkg.add("TrajectoryOptimization")
```

# What's New
`TrajectoryOptimization.jl` underwent significant changes between versions `v0.1` and `v0.2`. The new code is significantly faster (up to 100x faster). The core part of the ALTRO solver (everything except the projected newton phase) is completely allocation-free once the solver has been initialized. Most of the API has changed significantly. See the documentation for more information on the new API.

In `v0.3` the package was split into several different packages for increased modularity. These include [RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl), [Altro.jl](https://github.com/RoboticExplorationLab/Altro.jl), [RobotZoo.jl](https://github.com/bjack205/RobotZoo.jl), and [TrajOptPlots.jl](https://github.com/RoboticExplorationLab/TrajOptPlots.jl).

## Quick Start
To run a simple example of a constrained 1D block move see script in [`/examples/quickstart.jl`](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl/blob/master/examples/quickstart.jl).

## Examples
Notebooks with more detailed examples can be found [here](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl/tree/master/examples)

## Related Papers
* [IROS 2019 paper](https://rexlab.stanford.edu/papers/altro-iros.pdf).
