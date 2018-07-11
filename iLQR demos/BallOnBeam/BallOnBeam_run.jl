using BallOnBeam
using iLQR

n = 4
m = 1

# costs
Q = 5e-4*eye(n)
Qf = 500.*eye(n)
R = 1e-5*eye(m)

# simulation
tf = 1.
dt = 0.01

# initial and goal states
x0 = [0.1;0.;0.;0.]
xf = [0.5;0.;0.;0.]

# Objects
model = iLQR.Model(BallOnBeam.dynamics, n, m)
obj = iLQR.Objective(Q, R, Qf, tf, x0, xf)
solver = iLQR.Solver(model, obj, dt)

# @show solver.fd(rand(4), rand(1))
@time X, U = iLQR.solve(solver, iterations=10)
