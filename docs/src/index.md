# TrajectoryOptimization.jl

```@meta
CurrentModule = TrajectoryOptimization
```

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

The primary focus of this package is developing the ALTRO algorithm, although we hope this will extend to many algorithms in the future.


# Getting Started
To set up and solve a trajectory optimization problem with TrajectoryOptimization.jl, the user will go through the following steps:

1) Create a [Model](@ref)
2) Create a [CostFunction](@ref)
3) Instantiate a [Problem](@ref) with constraints
4) Pick an appropriate solver
5) Solve the problem
6) Analyze the solution


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
While the model defines the dynamics of the system, the Objective defines what you want the dynamics to do. The Objective class defines the objective function via a [`CostFunction`](@ref) type, as well as the initial states and trajectory duration. The Objective class also specifies constraints on the states and controls. Both iLQR and Direct Collocation (DIRCOL) allow generic cost functions of the form ``g(x,u) \leq 0`` or ``h(x,u) = 0``: any generic function of the state and control is permitted, but no couples between time steps is allowed.

### Creating a Cost Function
The cost (or objective) function is the first piece of the objective. While the majority of trajectory optimization problems have quadratic objectives, TrjaectoryOptimization.jl allows the user to specify any generic cost function of the form ``\ell_N(x_N) + \sum_{k=0}^N \ell(x_k,u_k)``. Currently GenericObjective is only supported by iLQR, and not by DIRCOL. Since iLQR relies on 2nd Order Taylor Series Expansions of the cost, the user may specify analytical functions for this expansion in order to increase performance; if the user does not specify an analytical expansion it will be generated using ForwardDiff.

```@docs
QuadraticCost
LQRCost
GenericCost
```

### Creating the Objective
Once the cost function is specified, the user then creates either an Unconstrained or Constrained Objective. When running iLQR, specifying any ConstrainedObjective will perform outer loop updates using an Augmented Lagrangian method.

```@docs
UnconstrainedObjective
ConstrainedObjective
LQRObjective
```

Since objectives are immutable types, the user can "update" the objective using the following function
```@docs
update_objective
```

## Solving the Problem
With a defined model and objective, the next step is to create a [`Solver`](@ref) type. The Solver is responsible for storing solve-dependent variables (such as number of knot points, step size, discrete dynamics functions, etc.) and storing parameters used during the solve (via [`SolverOptions`](@ref)). The solver contains both the Model and Objective and contains all information needed for the solve, except for the initial trajectories.

```@docs
Solver
```

Once the solver is created, the user must create an initial guess for the control trajectory, and optionally a state trajectory. For simple problems a initialization of random values, ones, or zeros works well. For more complicated systems it is usually recommended to feed trim conditions, i.e. controls that maintain the initial state values. For convenience, the function [`get_sizes`](@ref) returns n,m,N from the solver. Note that for trajectory optimization the control trajectory should be length N-1 since there are no controls at the final time step. However, DIRCOL uses controls at the final time step, and iLQR will simply discard any controls at the time step. Therefore, an initial control trajectory of size (m,N) is valid (but be aware that iLQR will return the correctly-sized control trajectory). Once the initial state and control trajectories are specified, they are passed with the solver to one of the [`solve`](@ref) methods.

## Solve Methods
With a Solver instantiated, the user can then choose to solve the problem using iLQR (`solve` function) or DIRCOL (`solve_dircol` function), where are detailed below

### iLQR Methods
#### Unconstrained Problem
For unconstrained problems the user doesn't have any options. iLQR can usually solve unconstrained problems without any modification. Simply call the `solve` method, passing in a initial guess for the control trajectory:
```
solve(solver,U0)
```
where `U0` is a Matrix of size `(m,N-1)` (although a trajectory of N points will also be accepted).

### Constrained Problem
The default constrained iLQR method uses an Augmented Lagrangian approach to handle the constraints. Nearly all of the options in [SolverOptions](@ref) determine parameters used by the Augmented Lagrangian method. Other than now having more parameters to tune for better performance (see another section for tips), the user solves a constrained problem using the exact same method for solving an unconstrained problem.

### Constrained Problem with Infeasible Start
One of the primary disadvantages of iLQR (and most indirect methods) is that the user must specify an initial input trajectory. Specifying a good initial guess can often be difficult in practice, whereas specifying a guess for the state trajectory is typically more straightforward. To overcome this limitation, TrajectoryOptimization adds artificial controls to the discrete dynamics $$x_{k+1} = f_d(x_k,u_k) + \diag{(\tidle{u}_1,\hdots,\tidle{u}_n)} such that the system is fully-actuated (technically over-actuated), so that an arbitrary state trajectory can be achieved. These artificial controls are then constrained to be zero using the Augmented Lagrangian method. This results in an algorithm similar to that of DIRCOL: initial solutions are dynamically infeasible but become dynamically infeasible at convergence. To solve the problem using "infeasible start", simply pass in an initial guess for the state and control:
```
solve(solver,X0,U0)
```

## DIRCOL Method
Problems can be solved using DIRCOL by simply calling
```
solve_dircol(solver,X0,U0)
```
