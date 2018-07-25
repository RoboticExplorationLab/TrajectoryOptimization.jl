include("../iLQR.jl")
using iLQR
using RigidBodyDynamics
using ForwardDiff
using Plots
using MeshCatMechanisms

# Acrobot
urdf = ".//urdf//doublependulum.urdf"
isfile(urdf)
acrobot = parse_urdf(Float64,urdf)
state = MechanismState(acrobot)

n = 4
m = 1

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

set_configuration!(state, x0[1:2])
set_velocity!(state, x0[3:4])

# costs
Q = 1e-4*eye(n)
Qf = 250.0*eye(n)
R = 1e-4*eye(m)

# simulation
tf = 5.0
dt = 0.01

model = iLQR.Model(acrobot, [false,true])
obj = iLQR.Objective(Q,R,Qf,tf,x0,xf)
solver = iLQR.Solver(model, obj, dt=dt)

U0 = ones(m, solver.N-1)*5
X, U = iLQR.solve(solver, U0)
X_sr, U_sr = iLQR.solve_sqrt(solver, U0)

P = plot(linspace(0,tf,size(X,2)),X[1,:],title="Acrobot",label="\Theta")
P = plot!(linspace(0,tf,size(X,2)),X[2,:],ylabel="State",label="\dot{\Theta}")
