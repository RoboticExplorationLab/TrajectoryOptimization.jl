# TrajectoryOptimization

![Build Status](https://travis-ci.org/RoboticExplorationLab/TrajectoryOptimization.jl.svg?branch=master)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/dev)

A library of solvers for trajectory optimization problems written in Julia. Currently, the following methods are implemented with a common interface:

[ALTRO (Augmented Lagrangian TRajectory Optimizer)](https://rexlab.stanford.edu/papers/altro-iros.pdf): A novel fast solver for trajectory optimization that features:
  * General nonlinear cost functions, including minimum time problems
  * General nonlinear state and input constraints
  * Infeasible initialization
  * Square-root methods for improved numerical conditioning

Direct Collocation (DIRCOL)
  * Interfaces to [IPOPT](https://github.com/coin-or/Ipopt) and [SNOPT](http://www.sbsi-sol-optimize.com/asp/sol_products_snopt_desc.htm) solvers

All methods utilize Julia's extensive autodifferentiation capabilities via [ForwardDiff.jl](http://www.juliadiff.org/ForwardDiff.jl/) so that the user does not need to specify derivatives of cost or constraint functions, and dynamics can be computed directly from a URDF file via [RigidBodyDynamics.jl](https://github.com/JuliaRobotics/RigidBodyDynamics.jl).
