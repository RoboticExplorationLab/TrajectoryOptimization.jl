# Setting up a Problem
```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
```

The [Problem](@ref) type represents the trajectory optimization problem to be solved, which consists of the following information:

* Dynamics model: the physical system that is being controlled, specified by a system of difference equations. We assume that continuous dynamics have been discretized, see [From Continuous Model](@ref).
* Cost function: the function to be minimized, and must be of the form ``\ell_N(x_N) + \sum_{k=0}^N \ell(x_k,u_k)``
* Constraints: Aside from the dynamics constraints imposed by the dynamics model, the problem specifies "stage" constraints of the form ``c(x_k,u_k)`` or "terminal" constraints of the form ``c(x_N)``.
* Initial state: all trajectory optimization algorithms require the initial state.
* N: the number of knot points (or number of discretization points)
* dt: the time step used for integrating the continuous dynamics, if required by the dynamics model

The [Problem](@ref) type also stores the state and control input trajectories (i.e. the primal variables).

## Creating a Problem
A problem is typically created using the following constructors

```@docs
Problem(model::Model,cost::CostFunction,x0::Vector{T},U::VectorTrajectory{T},dt::T) where T
Problem(model::Model,cost::CostFunction,x0::Vector{T},U::Matrix{T},dt::T) where T
```
where `U` is either an n × N-1 matrix or a vector of n-dimensional vectors (VectorTrajectory). A bare-minimum constructor is also available

```@docs
Problem(model::Model,cost::CostFunction,N::Int,dt::T) where T
```
which initializes the initial state and the control input trajectory to zeros.

The inner constructor can also be used with caution.
```@docs
Problem(model::Model, cost::CostFunction, constraints::ConstraintSet,
      x0::Vector{T}, X::VectorTrajectory, U::VectorTrajectory, N::Int, dt::T) where T
```

## Adding constraints
A constraint can be added once the problem is created using

```@docs
add_constraints
```
