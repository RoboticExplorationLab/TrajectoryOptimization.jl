using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Plots
const TO = TrajectoryOptimization
import TrajectoryOptimization: dynamics, RK3, AbstractModel, KnotPoint, Traj, StaticBoundConstraint, GoalConstraint,
    ConstraintVals, ConstraintSets, StaticProblem, StaticALSolver
import TrajectoryOptimization: dynamics

# Define dynamics model
#   must inherit from AbstractModel
struct Cartpole{T} <: AbstractModel
    mc::T
    mp::T
    l::T
    g::T
end

Cartpole() = Cartpole(1.0, 0.2, 0.5, 9.81)

#   must define this
Base.size(::Cartpole) = 4,1

#   must define this
function dynamics(model::Cartpole, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end

# Instantiate dynamics model
model = Dynamics.Cartpole()
n,m = size(model)

# Discretization info
tf = 5.0  # final time
N = 101   # number of knot points
dt = tf / (N-1)

# Define initial and final states (be sure to use Static Vectors!)
x0 = @SVector [0, 0.,0., 0.]
xf = @SVector [0., pi, 0., 0.] # (ie, swing up)

# Define a quadratic cost
Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N)


# Constraints
#   bound constraint
u_bnd = 3.0
bnd = StaticBoundConstraint(n,m, u_min=-u_bnd*(@SVector ones(m)), u_max=u_bnd*(@SVector ones(m)))

#   goal constraint
goal = GoalConstraint(SVector{n}(xf))

#   create constraint value types (pre-allocates arrays for values, Jacobians, active set, etc.)
#   first argument is the constraint, second is the range of knot points to which the constraint is applied
#   TODO: do this automatically when building ConstraintSets
con_bnd = ConstraintVals(bnd, 1:N-1)
con_goal = ConstraintVals(goal, N:N)

#   create constraint set for problem
conSet = ConstraintSets([con_bnd, con_goal], N)

# Define the initial trajectory
#  set initial states the NaNs since it will overwitten TODO: do this automatically
u0 = @SVector [0.01]
U0 = [copy(u0) for k = 1:N]
xs = NaN*@SVector zeros(n)
us = SVector{m}(u0)
Z = [KnotPoint(xs,us,dt) for k = 1:N]
Z[end] = KnotPoint(xs,m)

# Build problem
#   TODO: better constructors
prob = StaticProblem(model, obj, conSet, x0, xf,
    deepcopy(Z), deepcopy(Z), N, dt, dt*(N-1))

# Build the solver
opts = StaticALSolverOptions{Float64}()
solver = StaticALSolver(prob, opts)

# Convert to Augmented Lagrangian problem
#   TODO: convert this automatically (will incur allocations)
prob_al = TO.convertProblem(prob, solver)

# Solve
#   reset the control before the solve so you can re-run it easily
initial_controls!(prob_al, u0)
solve!(prob_al, solver)


# Analyze results
#   look in solver.stats for stats on the solve
println("Outer loop iterations: ", solver.stats.iterations)
#   get final constraint violation
println("Max violation: ", max_violation(prob_al))
#   get final cost
println("Final cost: ", cost(prob_al))
#   extract trajectories and plot
X = state(prob_al)
U = control(prob_al)
plot(X, 1:2)
plot(U)
#   plot cost convergence
plot(1:solver.stats.iterations, solver.stats.cost, yscale=:log10)
#   plot constraint violation
plot(1:solver.stats.iterations, solver.stats.c_max, yscale=:log10)

# Benchmark the result
@btime begin
    initial_controls!(prob_al, U0)
    solve!(prob_al, solver)
end

#   there shouldn't be any allocations
initial_controls!(prob_al, U0)
@allocated solve!(prob_al, solver)
