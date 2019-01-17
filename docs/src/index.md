# TrajectoryOptimization.jl

Documentation for TrajectoryOptimization.jl

```@contents
```

# Overview
The purpose of this package is to provide a testbed for state-of-the-art trajectory optimization algorithms. In general, this package focuses on trajectory optimization problems of the form
(put LaTeX here)

This package currently implements both indirect and direct methods for trajectory optimization:
* Iterative LQR (iLQR): indirect method based on differential dynamic programming
* Direct Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solver

The primary focus of this package is developing the iLQR algorithm, although we hope this will extend to many algorithms in the future.


# Getting Started
In order to set up a trajectory optimization problem, the user needs to create a Model and Objective

## Creating a Model
There are two ways of creating a model:
1) An in-place analytic function of the form f(ẋ,x,u)
2) A URDF

### Analytic Models
To create an analytic model create an in-place function for the continuous dynamics. The function must be of the form
`f(ẋ,x,u)`
where ẋ ∈ Rⁿ is the state derivative vector, x ∈ Rⁿ is the state vector, and u ∈ Rᵐ is the control input vector. The function should not return any values, but should write ẋ "inplace," e.g. `ẋ[1] = x[2]*u[2]` NOT `ẋ = f(x,u)`. This makes a significant difference in performance.

Specifying discrete-time dynamics directly is currently not supported (but should be straight-forward to implement).

The Model type is then created using the following signature:
`model = Model(f,n,m)` where `n` is the dimension of the state input and `m` is the dimension of the control input.

```@docs
Model(f::Function, n::Int, m::Int)
```

### URDF Model
This package relies on RigidBodyDynamics.jl to parse URDFs and generate dynamics functions for them. There are several useful constructors:

```@docs
Model(mech::Mechanism)
Model(mech::Mechanism, torques::Array)
Model(urdf::String)
Model(urdf::String, torques::Array)
```
