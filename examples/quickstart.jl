using TrajectoryOptimization
# using Altro
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

## Problem definition

# Model and discretization
model = DoubleIntegrator(1.0)
n,m = RD.dims(model)
tf = 3.0         # final time (sec)
N = 21           # number of knot points
dt = tf / (N-1)  # time step (sec)
@test (n,m) == (4,2)

# Objective
x0 = SA[0,0.,0,0]  # initial state
xf = SA[0,2.,0,0]  # final state

Q = Diagonal(@SVector ones(n))
R = Diagonal(@SVector ones(m))
Qf = Q*(N-1)
obj = LQRObjective(Q, R, Qf, xf, N)

# Constraints
# * Adds the following constraints
#   - Constraint to get to the goal
#   - A single circular obstacle at (0,1.0) with radius 0.5
#   - Norm constraint on the controls: ||u||₂ <= 5.0
#   - Bounds the controls between -10 and 10
# * Note that the time indices are chosen to avoid redundant constraints at the 
#   first and last time steps.
cons = ConstraintList(n,m,N)
add_constraint!(cons, GoalConstraint(xf), N)
add_constraint!(cons, CircleConstraint(n, SA[0.0], SA[1.0], SA[0.5]), 2:N-1)
add_constraint!(cons, NormConstraint(n, m, 5.0, TO.SecondOrderCone(), :control), 1:N-1)
add_constraint!(cons, BoundConstraint(n,m, u_min=-10, u_max=10), 1:N-1)

# Create problem
prob = Problem(model, obj, x0, tf, xf=xf, constraints=cons)

## Methods on Problem
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
# u1 = controls(prob, 1)

## Constraints
# `ConstraintList` supports indexing (and iteration):
constraints = get_constraints(prob)
goalcon = constraints[1]
circlecon = constraints[2]
normcon = constraints[3]

# Can use the `zip` method to iterate over the constraint and its assigned indices
for (inds,con) in zip(constraints)
    @show inds
end

# Get the cone for each constraint
#  `ZeroCone` is an equality constraint
#  `NegativeOrthant` is an inequality constraint c(x) <= 0
cones = map(TO.sense, constraints)
@test cones == [TO.ZeroCone(), TO.NegativeOrthant(), TO.SecondOrderCone(), TO.NegativeOrthant()]

# Get bounds for each constraint
# * These are useful when passing to interfaces like MathOptInterface that expect 
#   a vector of constraints and their upper and lower bounds
# * Equality constraints have upper and lower bounds of zero
# * Inequality constraints have an upper bound of zero 
# * If a constraint can be represented as a simple bound on the states and/or controls,
#   e.g. `xmin ≤ x ≤ xmax` or `xumin ≤ u ≤ umax` the `TO.is_bound` method will return `true``.
is_bound_constraint = map(TO.is_bound, constraints)
@test is_bound_constraint == [true, false, false, true]

lower_bounds = vcat(Vector.(map(TO.lower_bound, constraints))...)
upper_bounds = vcat(Vector.(map(TO.upper_bound, constraints))...)
@test lower_bounds ≈ [zeros(n); fill(-Inf, 1); fill(-Inf, m+1); fill(-Inf, n); fill(-10, m)]
@test upper_bounds ≈ [zeros(n); fill(0.0, 1); fill(+Inf, m+1); fill(+Inf, n); fill(+10, m)]

# Constraint evaluation
# * Each constraint inherits from the `AbstractFunction` interface defined in RobotDynamics
# * They can be individually evaluated using the methods provided by that interface
# * TrajectoryOptimization also provides some methods for evaluating the constraints 
#   for multiple time steps

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