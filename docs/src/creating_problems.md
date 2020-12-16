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
  \min_{x_{0:N},u_{0:N-1}} \quad & (x_N-x_f)^T Q_f (x_N-x_f) + dt \sum_{k=0}^{N-1} (x_k-x_f)^T Q (x_k - x_f) + u^T R u  \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & |u_k| \leq 3 \\
                                 & x_N = x_f \\
\end{aligned}
```
We'll quickly set up the dynamics, objective, and constraints. See previous sections for more
details on how to do this.


```@example
using TrajectoryOptimization
using RobotZoo: Cartpole
using StaticArrays, LinearAlgebra

# Dynamics and Constants
model = Cartpole()
n,m = size(model)
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
prob = Problem(model, obj, xf, tf, constraints=conSet, x0=x0, integration=RK3)
```
where the keyword arguments are, of course, optional.

This constructor has the following arguments:
* (required) `model::AbstractModel` - dynamics model
* (required) `obj::AbstractObjective` - objective function
* (required) `xf::AbstractVector` - goal state (this will be made optional in the near future)
* (required) `tf::AbstractFloat` - final time
* (optional) `constraints::ConstraintSet` - constraint set. Default is no constraints.
* (optional) `x0::AbstractVector` - Initial state. Default is the zero vector.
* (optional) `N::Int` - number of knot points. Default is given by length of objective.
* (optional) `dt::AbstractFloat` - Time step length. Can be either a scalar or a vector of length `N`. Default is calculated using `tf` and `N`.
* (optional) `integration::Type{<:QuadratureRule}` - Quadrature rule for discretizing the dynamics. Default is given by `TrajectoryOptimization.DEFAULT_Q`.
* (optional) `X0` - Initial guess for state trajectory. Can either be a matrix of size `(n,N)` or a vector of length `N` of `n`-dimensional vectors.
* (optional) `U0` - Initial guess for control trajectory. Can either be a matrix of size `(m,N)` or a vector of length `N-1` of `n`-dimensional vectors.


## Initialization
A good initialization is critical to getting good results for nonlinear optimization problems.
TrajectoryOptimization.jl current supports initialization of the state and control trajectories.
Initialization of dual variables (i.e. Lagrange multipliers) is not yet support but will be
included in the near future. The state and control trajectories can be initialized directly
in the constructor using the `X0` and `U0` keyword arguments described above, or using the
following methods:

```julia
initial_states!(prob, X0)
initial_controls!(prob, U0)
```
where, again, these can either be matrices or vectors of vectors of the appropriate size.
It should be noted that these methods work on either `Problem`s or instances of `AbstractSolver`.

Alternatively, the problem can be initialized with both the state and control trajectories
simultaneously by passing in a vector of `KnotPoint`s, described in the next sections.

## `KnotPoint` Type
Internally, TrajectoryOptimization.jl stores the state and controls at each time step as a
concatenated vector inside the `KnotPoint` type defined by RobotDynamics.jl.
In addition to storing the state and control, the `KnotPoint` type also stores the
time and time step length for the current knot point. See the documention in RobotDynamics
for more information.

## `Traj` Type
The `Traj` type is simply a vector of `KnotPoint`s. However, it provides a few helpful methods
for constructing and working vectors of `KnotPoint`s, which effectively describe a discrete-time
state-control trajectory.

```@docs
Traj
```

### Other Methods
You can extract the state and control trajectories separately with the following methods:
```julia
states(Z::Traj)
controls(Z::Traj)
```
Note that these methods also work on `Problem`. 

The states, control, and time trajectories can be set independently with the following methods:
```julia
set_states!(Z::Traj, X::Vector{<:AbstractVector})
set_controls!(Z::Traj, U::Vector{<:AbstractVector})
set_times!(Z::Traj, t::Vector)
```

To initialize a problem with a given `Traj` type, you can use
```
initial_trajectory!(::Problem, Z::AbstractTrajectory)
```
