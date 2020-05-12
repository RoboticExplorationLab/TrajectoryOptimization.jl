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
structure that allows purpose-built solvers such as [ALTRO.jl](https://github.com/bjack205/ALTRO.jl)
to gain significant computational savings over the use of more generalized NLP solvers such
as SNOPT and Ipopt.

Trajectory optimization problems are of the form,
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
* ForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints
* Efficient methods for evaluating the trajectory optimization problem as a general NLP,
  so that it can be passed off to NLP solvers such as Ipopt or SNOPT via
  [MathOptInterface.jl](https://github.com/JuliaOpt/MathOptInterface.jl).
