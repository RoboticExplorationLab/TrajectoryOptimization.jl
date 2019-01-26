# TrajectoryOptimization

![Build Status](https://travis-ci.org/RoboticExplorationLab/TrajectoryOptimization.jl.svg?branch=master)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/TrajectoryOptimization.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://RoboticExplorationLab.github.io/TrajectoryOptimization.jl/dev)

A package for solving trajectory optimization problems in Julia.

This package solves optimal control problems using:
* Iterative LQR (iLQR)
  * Augmented Lagrangian method for nonlinear constraints
  * Infeasible state trajectory initialization
  * Minimum time
  * Square root method for numerical conditioning
  * Nonlinear cost functions

* Direct Collocation (DIRCOL)
  * IPOPT, SNOPT

-Autodifferentiation of nonlinear dynamics and constraint functions via ForwardDiff.jl

-Dynamics directly from a URDF via RigidBodyDynamics.jl
