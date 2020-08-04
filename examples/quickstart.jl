using TrajectoryOptimization
using Altro
using RobotDynamics
using StaticArrays
using LinearAlgebra
using RobotDynamics
using Altro
const TO = TrajectoryOptimization

struct DoubleIntegrator{T} <: TO.AbstractModel
    mass::T
end

function RobotDynamics.dynamics(model::DoubleIntegrator, x, u)
    SA[x[2], u[1] / model.mass]
end

Base.size(::DoubleIntegrator) = 2,1

# Model and discretization
model = DoubleIntegrator(1.0)
n,m = size(model)
tf = 3.0  # sec
N = 21    # number of knot points

# Objective
x0 = SA[0,0.]  # initial state
xf = SA[1,0.]  # final state

Q = Diagonal(@SVector ones(n))
R = Diagonal(@SVector ones(m))
obj = LQRObjective(Q, R, N*Q, xf, N)

# Constraints
cons = ConstraintList(n,m,N)
add_constraint!(cons, GoalConstraint(xf), N)
add_constraint!(cons, BoundConstraint(n,m, u_min=-10, u_max=10), 1:N-1)

# Create and solve problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
solver = ALTROSolver(prob)
cost(solver)           # initial cost
solve!(solver)         # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# Get the state and control trajectories
X = states(solver)
U = controls(solver)
