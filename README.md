# TrajectoryOptimization

![Build Status](https://travis-ci.org/RoboticExplorationLab/TrajectoryOptimization.jl.svg?branch=master)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/dev)

A package for solving constrained and unconstrained trajectory optimization problems in Julia.

The package currently has the following solvers:
* iLQR
* Direct Collocation

The package currently has the following capabilities:
* Solve trajectory optimization problems with quadratic costs and nonlinear equality or inequality constraints (both stage and terminal)
* Auto-differentiation of non-linear dynamics and constraint functions via ForwardDiff.jl
* Generate dynamics directly from a URDF via RigidBodyDynamics.jl
* Initialize the solver with an infeasible trajectory, allowing the user to input an initial guess for either states, controls, or both.
