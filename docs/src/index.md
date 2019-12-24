# TrajectoryOptimization.jl Documentation

```@meta
CurrentModule = TrajectoryOptimization
```

Documentation for TrajectoryOptimization.jl

```@contents
Pages = ["index.md"]
```


## Overview
This package is a testbed for state-of-the-art trajectory optimization algorithms. Trajectory optimization problems are of the form,
```math
\begin{aligned}
  \min_{x_{0:N},u_{0:N-1}} \quad & \ell_f(x_N) + \sum_{k=0}^{N-1} \ell_k(x_k, u_k, dt) \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & g_k(x_k,u_k) \leq 0, \\
                                 & h_k(x_k,u_k) = 0.
\end{aligned}
```

This package currently implements the following methods for solving trajectory optimization problems:
* Iterative LQR (iLQR): indirect method based on Differential Dynamic Programming
* AL-iLQR: iLQR within an Augmented Lagrangian framework
* Direct Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solver
* ALTRO (Augmented Lagrangian Trajectory Optimizer): A novel algorithm developed by the Robotic Exploration Lab at Stanford University, which uses iLQR within an augmented Lagrangian framework combined with a "Projected Newton" direct method for solution polishing and enforcement of feasibility.

Key features include:
* Support for general, per-timestep constraints
* ForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints
* URDF parsing via [RigidBodyDynamics]


## Getting Started
To set up and solve a trajectory optimization problem with `TrajectoryOptimization.jl`, the user will go through the following steps:

1) Create a [Model](@ref model_section)
2) Create an [Objective](@ref objective_section)
3) (Optionally) Add [constraints](@ref constraint_section)
4) Instantiate a [Problem](@ref problem_section)
5) Select a [solver](@ref solver_section)
6) Solve the problem
