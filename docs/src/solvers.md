# Solvers
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["solvers.md"]
```

# Iterative LQR (iLQR)
iLQR is an unconstrained indirect method for trajectory optimization that parameterizes only the controls and enforces strict dynamics feasibility at every iteration by simulating forward the dynamics with an LQR feedback controller. The main algorithm consists of two parts: 1) a backward pass that uses Differential Dynamic Programming to compute recursively a quadratic approximation of the cost-to-go, along with linear feedback and feed-forward gain matrices, `K` and `d`, respectively, for an LQR tracking controller, and 2) a forward pass that uses the gains `K` and `d` to simulate forward the full nonlinear dynamics with feedback.

The iLQR solver has the following solver options
```@docs
iLQRSolverOptions
```

# Augmented Lagrangian
Augmented Lagrangian
```@docs
AugmentedLagrangianSolverOptions
```

# ALTRO
ALTRO
```@docs
ALTROSolverOptions
```

# Direct Collocation (DIRCOL)
