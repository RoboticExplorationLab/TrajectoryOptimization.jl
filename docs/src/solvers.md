# Solvers
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
```

# Iterative LQR (iLQR)
iLQR is an indirect method for trajectory optimization that parameterizes only the controls and enforces strict dynamics feasibility at every iteration by simulating forward the dynamics with an LQR feedback controller. The main algorithm consists of two parts: 1) a backward pass that uses differential dynamic programming to compute recursively a quadratic approximation of the cost-to-go, along with linear feedback and feed-forward gain matrices, `K` and `d`, respectively, for an LQR tracking controller, and 2) a forward pass that uses the gains `K` and `d` to simulate forward the dynamics with feedback.

The vanilla iLQR algorithm is incapable of handling constraints aside from the dynamics. Any reference to the iLQR algorithm within TrajectoryOptimization.jl will assume the problem is solving an unconstrained problem. Other algorithms, such as ALTRO, use iLQR an an internal, unconstrained solver to solve a trajectory optimization problem with constraints.

The iLQR solver has the following solver options
```@docs
iLQRSolverOptions
```

Augmented Lagrangian
```@docs
AugmentedLagrangianSolverOptions
```
