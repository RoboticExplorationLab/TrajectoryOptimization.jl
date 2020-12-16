# TrajectoryOptimization.jl Documentation

```@meta
CurrentModule = TrajectoryOptimization
```

Documentation for TrajectoryOptimization.jl

```@contents
Pages = ["index.md"]
```


## Overview
This package facilitates the definition and evaluation of trajectory optimization problems.
Importantly, this package should be considered more of a modeling framework than an
optimization solver, similar to [Convex.jl](https://github.com/JuliaOpt/Convex.jl).
While general trajectory optimization problems are nonconvex, primarily due to the
presence of nonlinear equality constraints imposed by the dynamics, they exhibit a unique
structure that allows purpose-built solvers such as [Altro.jl](https://github.com/bjack205/ALTRO.jl)
to gain significant computational savings over the use of more generalized NLP solvers such
as SNOPT and Ipopt.

This package deals with trajectory optimization problems of the form,
```math
\begin{aligned}
  \min_{x_{0:N},u_{0:N-1}} \quad & \ell_f(x_N) + \sum_{k=0}^{N-1} \ell_k(x_k, u_k, dt) \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & g_k(x_k,u_k) \leq 0, \\
                                 & h_k(x_k,u_k) = 0.
\end{aligned}
```

Key features include:
* Easy and intuitive interface for setting up trajectory optimization problems
* Support for general, per-timestep constraints
* Support for Second-Order Cone constraints 
* ForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints
* Efficient methods for evaluating the trajectory optimization problem as a general NLP, so that it can be passed off to NLP solvers such as Ipopt or SNOPT via
  [MathOptInterface.jl](https://github.com/JuliaOpt/MathOptInterface.jl).

## Installation
TrajectoryOptimization.jl can be installed via the Julia package manager. Within the Julia
REPL:
```
] # activate the package manager
(v1.5) pkg> add TrajectoryOptimization
```
A specific version can be specified using
```
(v1.5) pkg> add TrajectoryOptimization@0.4.1
```
Or you can check out the master branch with
```
(v1.5) pkg> add TrajectoryOptimization#master
```
Lastly, if you want to clone the repo into your `.julia/dev/` directory for development, you can use
```
(v1.5) pkg> dev TrajectoryOptimization
```

This will automatically add all package dependencies (see [`Project.toml`](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl/blob/master/Project.toml)).
If you want to explicitly use any of these dependencies (such as [RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl)), 
you'll need to individually add those packages to your environment via the package manager.