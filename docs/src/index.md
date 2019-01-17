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

Key features include the use of ForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints; the use of RigidBodyDynamics to work directly from URDF files; and the ability to specify general constraints. 

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

## Creating an Objective
While the model defines the dynamics of the system, the Objective defines what you want the dynamics to do. The Objective class defines the objective function via a (CostFunction)@ref type, as well as the initial states and trajectory duration. The Objective class also specifies constraints on the states and controls. Both iLQR and Direct Collocation (DIRCOL) allow generic cost functions of the form ``g(x,u) \\leq 0 or h(x,u) = 0``: any generic function of the state and control is permitted, but no couples between time steps is allowed.

### Creating a Cost Function
The cost (or objective) function is the first piece of the objective. While the majority of trajectory optimization problems have quadratic objectives, TrjaectoryOptimization.jl allows the user to specify any generic cost function of the form ``\\ell_N(x_N) + \\sum_{k=0}^N \\ell(x_k,u_k)``. Currently GenericObjective is only supported by iLQR, and not by DIRCOL. Since iLQR relies on 2nd Order Taylor Series Expansions of the cost, the user may specify analytical functions for this expansion in order to increase performance; if the user does not specify an analytical expansion it will be generated using ForwardDiff.

```@docs
QuadraticCost
LQRCost
GenericCost
```

### Creating the Objective
Once the cost function is specified, the user then creates either an Unconstrained or Constrained Objective. When running iLQR, specifying any ConstrainedObjective will perform outer loop updates using an Augmented Lagrangian method.
