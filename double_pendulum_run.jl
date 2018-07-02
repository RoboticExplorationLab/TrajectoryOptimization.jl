using DoublePendulum
using Plots

state = MechanismState(doublependulum)
# set_configuration!(state, [0., 0.])
# set_configuration!(vis, configuration(state))

## Problem Setup
n = 4
m = 2

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

set_configuration!(state, x0[1:2])
set_velocity!(state, x0[3:4])

# costs
Q = 1e-5*eye(n)
Qf = 25.*eye(n)
R = 1e-5*eye(m)

# simulation
tf = 1.
dt = 0.01

# objects
model = iLQR.Model(doublependulum)
obj = iLQR.Objective(Q, R, Qf, tf, x0, xf)
solver = iLQR.Solver(model, obj, f_jacobian, dt) # initialization
@time X, U = iLQR.solve(solver, iterations=10)
# solver = iLQR.Solver(model, obj, fx, fu, dt) # initialization

P = plot(linspace(0,tf,size(X,2)),X[1,:],title="Double Pendulum (iLQR and Rigid Body Dynamics)")
P = plot!(linspace(0,tf,size(X,2)),X[2,:],ylabel="State")
display(P)
