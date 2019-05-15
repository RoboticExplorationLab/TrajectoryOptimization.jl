# TrajectoryOptimization.jl

```@meta
CurrentModule = TrajectoryOptimization
```

Documentation for TrajectoryOptimization.jl

```@contents
```


# Overview
This package is a testbed for state-of-the-art trajectory optimization algorithms. Trajectory optimization problems are of the form,
(put LaTeX here)

This package currently implements both indirect and direct methods for trajectory optimization:
* Iterative LQR (iLQR): indirect method based on Differential Dynamic Programming
* Direct Collocation: direct method that formulates the problem as an NLP and passes the problem off to a commercial NLP solver

Key features include the use of ForwardDiff for fast auto-differentiation of dynamics, cost functions, and constraints; the use of RigidBodyDynamics to work directly from URDF files; and the ability to specify general constraints.

# Getting Started
To set up and solve a trajectory optimization problem with TrajectoryOptimization.jl, the user will go through the following steps:

1) Create a [Model](@ref)
2) Create an [Objective](@ref)
3) Instantiate a [Problem](@ref)
4) (Optionally) Add constraints
5) Select a solver
6) Solve the problem
7) Analyze the solution

## Creating a Model
There are two ways of creating a model:
1) from an in-place analytic function of the form f(y,x,u) that operates on y
2) from a URDF file

### Analytic Models
To create an analytic model, first create an in-place function for the continuous or discrete dynamics. The function must be of the form
`f(y,x,u)`
where y ∈ Rⁿ is the state derivative vector for continuous dynamics or the next state for discrete dynamics. x ∈ Rⁿ is the state vector, and u ∈ Rᵐ is the control input vector. The function should not return any values, but should write y "inplace," e.g. `y[1] = x[2]*u[2]` NOT `y = f(x,u)`. This makes a significant difference in performance.

The Model type is then created using the following signature:
`model = Model{D}(f,n,m)` where `n` is the dimension of the state input and `m` is the dimension of the control input, and D is a DynamicsType, either Continuous of Discrete.

```@docs
Model{D}(f::Function, n::Int, m::Int)
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
The [`Objective`](@ref) defines a metric for what you want the dynamics to do. The Objective type contains a CostFunctions for each stage of the trajectory.

### Creating a Cost Function
A [`CostFunction`](@ref) is required for each stage of the trajectory to define an Objective. While the majority of trajectory optimization problems have quadratic objectives, TrajectoryOptimization.jl allows the user to specify any generic cost function of the form ``\ell_N(x_N) + \sum_{k=0}^N \ell_k(x_k,u_k)``. Currently GenericObjective is only supported by iLQR, and not by DIRCOL. Since iLQR relies on 2nd Order Taylor Series Expansions of the cost, the user may specify analytical functions for this expansion in order to increase performance; if the user does not specify an analytical expansion it will be generated using ForwardDiff.

```@docs
QuadraticCost
LQRCost
GenericCost
```

## Solving the Problem
With a defined model and objective, the next step is to create a [`Problem`](@ref) type. The Problem contains both the Model and Objective and contains all information needed for the solve.

```@docs
Problem
```

Once the Problem is instantiated, the user must create an initial guess for the control trajectory, and optionally a state trajectory. For simple problems a initialization of random values, ones, or zeros works well. For more complicated systems it is usually recommended to feed trim conditions, i.e. controls that maintain the initial state values. Note that for trajectory optimization the control trajectory should be length N-1 since there are no controls at the final time step. However, DIRCOL uses controls at the final time step, and iLQR will simply discard any controls at the time step. Therefore, an initial control trajectory of size (m,N) is valid (but be aware that iLQR will return the correctly-sized control trajectory). Once the initial state and control trajectories are specified, they are passed with the solver to one of the [`solve`](@ref) methods.

## Solve Methods
With a Problem instantiated, the user can then select a solver: iLQR, AugmentedLagrangian, ALTRO, DIRCOL.

### Unconstrained Methods
iLQR is an unconstrained solver. For unconstrained Problems simply call the `solve` method:
```
solve(prob,iLQRSolverOptions())
```

### Constrained Methods
#### Augmented Lagrangian
The default constrained solver uses iLQR with an augmented Lagrangian framework to handle general nonlinear constraints. [AugmentedLagrangianSolverOptions](@ref) can be changed to effect solver performance. Other than now having more parameters to tune for better performance (see another section for tips), the user solves a constrained problem using the exact same method for solving an unconstrained problem.

#### ALTRO
### Constrained Problem with Infeasible Start
One of the primary disadvantages of iLQR (and most indirect methods) is that the user must specify an initial input trajectory. Specifying a good initial guess can often be difficult in practice, whereas specifying a guess for the state trajectory is typically more straightforward. To overcome this limitation, the ALTRO solver adds slack controls to the discrete dynamics $x_{k+1} = f_d(x_k,u_k) + \diag{(\tidle{u}_1,\hdots,\tidle{u}_n)}$ such that the system becomes artificially fully-actuated These slack controls are then constrained to be zero using the augmented Lagrangian method. This results in an algorithm similar to that of DIRCOL: initial solutions are dynamically infeasible but become dynamically infeasible at convergence. To solve the problem using "infeasible start", simply pass in an initial guess for the state and control:
```
copyto!(prob.X,X0)
solve(prob,ALTROSolverOptions())
```

### Minimum Time Problem
A minimum time problem can be solved using the ALTRO solver by setting tf=:min

## Direct Collocation (DIRCOL)
Problems can be solved using DIRCOL by simply calling
```
solve(prob,DIRCOLSolverOptions())
```
