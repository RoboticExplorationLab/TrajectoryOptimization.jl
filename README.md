# TrajectoryOptimization

![Build Status](https://travis-ci.org/RoboticExplorationLab/TrajectoryOptimization.jl.svg?branch=master)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl)

A package for solving constrained and unconstrained trajectory optimization problems in Julia.

The package currently has the following solvers:
* iLQR
  -general, nonlinear equality and inequality constraints for states and controls
  -infeasible start
  -minimum time
  -square root backward pass
* LQR
* Direct Collocation

Additionally:
* Autodifferentiation of nonlinear dynamics and constraint functions via ForwardDiff.jl
* Dynamics directly from URDF files via RigidBodyDynamics.jl
