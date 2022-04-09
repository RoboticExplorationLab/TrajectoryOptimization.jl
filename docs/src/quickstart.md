```@meta
CurrentModule = TrajectoryOptimization
```

# [Quickstart](@id quickstart_page)
This page provides a quick overview of the API to help you get started quickly. 
It also points to other places in the documentation to help you get more 
details.

## Step 1: Define the Dynamics Model
Define the dynamics model. See documentation for 
[RobotDynamics.jl](https://github.com/RoboticExplorationLab/RobotDynamics.jl) 
for more details.

```julia
using TrajectoryOptimization
using RobotDynamics
using StaticArrays
using LinearAlgebra
using RobotDynamics
using Test
const TO = TrajectoryOptimization
const RD = RobotDynamics

## Model Definition
struct DoubleIntegrator{T} <: RD.ContinuousDynamics 
    mass::T
end

function RD.dynamics(model::DoubleIntegrator, x, u)
    mass = model.mass
    vx,vy = x[3], x[4]
    ax,ay = u[1] / mass, u[2] / mass
    SA[vx, vy, ax, ay]
end

RD.state_dim(::DoubleIntegrator) = 4
RD.control_dim(::DoubleIntegrator) = 2
```

## Step 2: Define the Discretization
The next step is to instantiate our model, discretize it, if necessary, and 
define the number of knot points to use and time horizon.

```julia
model = DoubleIntegrator(1.0)
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
n,m = RD.dims(model)
tf = 3.0         # final time (sec)
N = 21           # number of knot points
dt = tf / (N-1)  # time step (sec)
```

## Step 3: Define the Objective
We now define our objective. See [Setting up an Objective](@ref objective_section)
[Cost Function Inteface](@ref cost_interface), and the [Cost/Objective API](@ref cost_api).
```julia
# Objective
x0 = SA[0,0.,0,0]  # initial state
xf = SA[0,2.,0,0]  # final state

Q = Diagonal(@SVector ones(n))
R = Diagonal(@SVector ones(m))
Qf = Q*(N-1)
obj = LQRObjective(Q, R, Qf, xf, N)
```

## Step 4: Define constraints
The next step is to define any constraints we want to add to our trajectory 
optimization problem. Here we define the following constraints:

- A constraint to get to the goal
- A single circular obstacle at (0,1.0) with radius 0.5
- Norm constraint on the controls: ||u||₂ <= 5.0
- Bounds the controls between -10 and 10

See [Creating Constraints](@ref constraint_section), [Constraint Interface](@ref),
and the [Constraint API](@ref constraint_api) for more details.

```julia
cons = ConstraintList(n,m,N)
add_constraint!(cons, GoalConstraint(xf), N)
add_constraint!(cons, CircleConstraint(n, SA[0.0], SA[1.0], SA[0.5]), 2:N-1)
add_constraint!(cons, NormConstraint(n, m, 5.0, TO.SecondOrderCone(), :control), 1:N-1)
add_constraint!(cons, BoundConstraint(n,m, u_min=-10, u_max=10), 1:N-1)
```
!!! note
    Note that the time indices are chosen to avoid redundant constraints at the 
    first and last time steps.

## Step 5: Create the Problem
The last step is to actually create the problem by passing all of the information 
we defined previously to the constructor. See 
[Setting up a Problem](@ref problem_section) and the [Problem API](@ref problem_api)
for more details.

```julia
# Create problem
prob = Problem(model, obj, x0, tf, xf=xf, constraints=cons)
```

## Step 6: Methods on Problems
The following code gives some examples of a few of the methods you can use on a 
[`Problem`](@ref) once it's created.

```julia
# Set initial controls
U0 = [randn(m) for k = 1:N]
initial_controls!(prob, U0)

# Set initial state
X0 = [zeros(n) for k = 1:N]
initial_states!(prob, X0)

# Evaluate the cost
J = cost(prob)
@test J ≈ mapreduce(+, 1:N) do k
    if k < N
        x,u = X0[k], U0[k]
        0.5 * (x - xf)'Q*(x - xf) + 0.5*u'R*u
    else
        x = X0[k]
        0.5 * (x - xf)'Qf*(x - xf)
    end
end

# Simulate the system forward
rollout!(prob)

# Extract states, controls, and times
X = states(prob)
U = controls(prob)
t = gettimes(prob)

Xrollout = [copy(x0) for k = 1:N]
for k = 1:N-1
    Xrollout[k+1] = RD.discrete_dynamics(
        get_model(prob, k), Xrollout[k], U0[k], dt*(k-1), dt
    )
end
@test Xrollout ≈ X 

# Convert to matrices
Xmat = hcat(Vector.(X)...)
Umat = hcat(Vector.(U)...)

# Extract out individual states and controls
x1 = states(prob, 2)
u1 = controls(prob, 1)
```

## Step 7: Working with Constraints
This section provides a few more details and example on how to work with the 
constraints we defined previously.

The [`ConstraintList`](@ref) type supports indexing and iteration:
```julia
constraints = get_constraints(prob)
goalcon = constraints[1]
circlecon = constraints[2]
normcon = constraints[3]
```

You can use the `zip` method to iterate over the constraint and its assigned 
indices:
```julia
for (inds,con) in zip(constraints)
    @show inds
end
```

You can identify the type of constraint using the 
[`TrajectoryOptimization.sense`](@ref) function, which returns a 
[`ConstraintSense`](@ref):
```julia
cones = map(TO.sense, constraints)
```

You can get the bounds of each constraint, which are useful when passing 
to interfaces like 
[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) 
that expect a vector of constraints with upper and lower bounds to encode 
equality/inequality constraints.

```julia

lower_bounds = vcat(Vector.(map(TO.lower_bound, constraints))...)
upper_bounds = vcat(Vector.(map(TO.upper_bound, constraints))...)
@test lower_bounds ≈ [zeros(n); fill(-Inf, 1); fill(-Inf, m+1); fill(-Inf, n); fill(-10, m)]
@test upper_bounds ≈ [zeros(n); fill(0.0, 1); fill(+Inf, m+1); fill(+Inf, n); fill(+10, m)]
```

You can use the [`is_bound`](@ref) function to check to see if the constraint can 
be represented as a simple bound constraint on the states or controls:
```julia
is_bound_constraint = map(TO.is_bound, constraints)
```

To evaluate constraints, you use the same interface from `AbstractFunction` 
defined in RobotDynamics:
```julia
# Evaluating a single constraint
# * Note that the `GoalConstraint` is only a function of the state, but
#   `NormConstraint` is a function of both the state and control
x,u = rand(model)  # generate some random states and controls
z = KnotPoint(x, u, 0.0, NaN)
c = zeros(RD.output_dim(goalcon))
RD.evaluate(goalcon, x)
RD.evaluate(goalcon, z)
RD.evaluate!(goalcon, c, x)
RD.evaluate!(goalcon, c, z)

c = zeros(RD.output_dim(normcon))
RD.evaluate(normcon, x, u)
RD.evaluate(normcon, z)
RD.evaluate!(normcon, c, x, u)
RD.evaluate!(normcon, c, z)

# Evaluating the Jacobian for a single constraint
# * Note that these methods require passing in a KnotPoint type
J = TO.gen_jacobian(goalcon)
c = zeros(RD.output_dim(goalcon))
RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), goalcon, J, c, z)

J = TO.gen_jacobian(normcon)
c = zeros(RD.output_dim(normcon))
RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), normcon, J, c, z)

# Evaluating constraint at multiple time steps 
Z = get_trajectory(prob)
inds = constraints.inds[1]
vals = [zeros(RD.output_dim(goalcon)) for i in inds] 
TO.evaluate_constraints!(RD.StaticReturn(), goalcon, vals, Z, inds)

inds = constraints.inds[2]
vals = [zeros(RD.output_dim(normcon)) for i in inds] 
TO.evaluate_constraints!(constraints.sigs[2], normcon, vals, Z, inds)

# Evaluating constraint Jacobians at multiple time steps
inds = constraints.inds[1]
jacs = [TO.gen_jacobian(goalcon) for i in inds] 
vals = [zeros(RD.output_dim(goalcon)) for i in inds] 
TO.constraint_jacobians!(
    TO.functionsignature(constraints, 1), TO.diffmethod(constraints, 1), 
    goalcon, jacs, vals, Z, inds
)

inds = constraints.inds[2]
jacs = [TO.gen_jacobian(normcon) for i in inds] 
vals = [zeros(RD.output_dim(normcon)) for i in inds] 
TO.constraint_jacobians!(
    TO.functionsignature(constraints, 2), TO.diffmethod(constraints, 2), 
    normcon, jacs, vals, Z, inds
)
```