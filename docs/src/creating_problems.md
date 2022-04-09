```@meta
CurrentModule = TrajectoryOptimization
```

# [4. Setting up a Problem](@id problem_section)
The [`Problem`](@ref) contains all of the information needed to solve a
trajectory optimization problem. At a minimum, this is the model, objective,
and initial condition. A `Problem` is passed to a solver, which extracts
needed information, and may or may or not modify its internal representation
of the problem in order to solve it (e.g. the Augmented Lagrangian solver
combines the constraints and objective into a single Augmented Lagrangian
objective.)

## Creating a Problem
Let's say we're trying to solve the following trajectory optimization problem:
```math
\begin{aligned}
  \min_{x_{0:N},u_{0:N-1}} \quad & (x_N-x_f)^T Q_f (x_N-x_f) + \sum_{k=0}^{N-1} (x_k-x_f)^T Q (x_k - x_f) + u^T R u  \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & |u_k| \leq 3 \\
                                 & x_N = x_f \\
\end{aligned}
```
We'll quickly set up the dynamics, objective, and constraints. See previous sections for more
details on how to do this.


```@example
using TrajectoryOptimization
using RobotDynamics
using RobotZoo: Cartpole
using StaticArrays, LinearAlgebra

# Dynamics and Constants
model = Cartpole()
n,m = RobotDynamics.dims(model)
N = 101   # number of knot points
tf = 5.0  # final time
x0 = @SVector [0, 0, 0, 0.]  # initial state
xf = @SVector [0, π, 0, 0.]  # goal state (i.e. swing up)

# Objective
Q = Diagonal(@SVector fill(1e-2,n))
R = Diagonal(@SVector fill(1e-1,m))
Qf = Diagonal(@SVector fill(100.,n))
obj = LQRObjective(Q, R, Qf, xf, N)

# Constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n,m, u_min=-3.0, u_max=3.0)
goal = GoalConstraint(xf)
add_constraint!(conSet, bnd, 1:N-1)
add_constraint!(conSet, goal, N:N)
```

The following method is the easiest way to set up a trajectory optimization problem:
```julia
prob = Problem(model, obj, x0, tf, constraints=conSet, xf=xf, integration=RK4)
```
where the keyword arguments are, of course, optional.

This constructor has the following arguments:
* (required) `model::RobotDynamics.AbstractModel` - dynamics model. Can also 
be a vector of `RobotDynamics.DiscreteDynamics` of length `N-1`.
* (required) `obj::AbstractObjective` - objective function
* (required) `x0::AbstractVector` - initial state 
* (required) `tf::AbstractFloat` - final time
* (optional) `constraints::ConstraintSet` - constraint set. Default is no constraints.
* (optional) `xf::AbstractVector` - Goal state.
* (optional) `N::Int` - number of knot points. Default is given by length of objective.
* (optional) `dt::AbstractFloat` - Time step length. Can be either a scalar or a vector of length `N-1`. Default is calculated using `tf` and `N`.
* (optional) `integration::RobotDynamics.QuadratureRule` - Quadrature rule for discretizing the dynamics. Default is given by `RobotDynamics.RK4`.
* (optional) `X0` - Initial guess for state trajectory. Can either be a matrix of size `(n,N)` or a vector of length `N` of `n`-dimensional vectors.
* (optional) `U0` - Initial guess for control trajectory. Can either be a matrix of size `(m,N)` or a vector of length `N-1` of `n`-dimensional vectors.


## Initialization
A good initialization is critical to getting good results for nonlinear optimization problems.
TrajectoryOptimization.jl current supports initialization of the state and control trajectories.
Initialization of dual variables (i.e. Lagrange multipliers) is not yet support but will be
included in the near future. The state and control trajectories can be initialized directly
in the constructor using the `X0` and `U0` keyword arguments described above, or using the
following methods:

- [`initial_states!`](@ref)
- [`initial_controls!`](@ref)

where, again, these can either be matrices or vectors of vectors of the appropriate size.
It should be noted that these methods work on either `Problem`s or instances of `AbstractSolver`.

Alternatively, the problem can be initialized with both the state and control trajectories by passing in a `RobotDynamics.SampledTrajectory` 
via [`initial_trajectory!`](@ref).

### Getters
The following methods can be used to get information out of the problem 
definition

- [`gettimes(::Problem)`](@ref)
- [`get_initial_time`](@ref)
- [`get_final_time`](@ref)
- [`get_constraints`](@ref)
- [`get_model`](@ref)
- [`get_objective`](@ref)
- [`get_trajectory`](@ref)
- [`get_initial_state`](@ref)
- [`get_final_state`](@ref)
- [`gettimes`](@ref)

To query the state and control dimensions along the trajectory, the preferred
method is to one of 

```julia
RobotDynamics.state_dim(prob, k)
RobotDynamics.control_dim(prob, k)
RobotDynamics.dims(prob, k)
```
which return `n`, `m`, or the tuple `n,m,N` for the state and control 
dimensions at time step `k`, and the horizon length `N`.

Alternatively, you can get the state and control dimensions as vectors of 
length `N` by omitting the time step `k`.

### Extracting states and controls
To extract the state and control trajectories, the `Problem` type supports
the same methods as `RobotDynamics.SampledTrajectory`, e.g.

```julia
states(prob)         # return a vector of state vectors
control(prob)        # return a vector of control vectors
states(prob, k)      # get a vector of the `k`th element of the state vector
controls(prob, i:j)  # get a vector the vectors of elements `i` through `j` of the controls vector. 
```

!!! tip
    To convert a vector of vectors to a 2D array, use:
    ```julia
    Xmat = hcat(Vector.(X)...)
    ```
    Note that converting to a vector is a safe way to avoid the expensive 
    operation of concatenating a bunch of static vectors, if the elements of 
    `X` are a subtype of `StaticArrays.StaticVector`.

### Other Methods
The cost of the current trajectory can be evaluated using [`cost`](@ref).
The initial state and the current controls trajectory can be used to simulate
the system forward (open-loop) to obtain a state trajectory via 
[`rollout!`](@ref).

For MPC applications, the following methods can be useful:
- `RobotDynamics.setinitialtime!(prob, t0)`
- [`set_initial_state!`](@ref)
- [`set_goal_state!`](@ref)
