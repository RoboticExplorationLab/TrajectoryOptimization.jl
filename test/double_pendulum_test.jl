include("..\\iLQR.jl")
using iLQR
using Plots
using Dynamics

## Double pendulum
urdf_dp = "urdf/doublependulum.urdf"
dp = iLQR.Model(urdf_dp)
n = dp.n
m = dp.m

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

# costs
Q = 0.0001*Diagonal{Float64}(I, 4)
Qf = 250.0*Diagonal{Float64}(I, 4)
R = 0.0001*Diagonal{Float64}(I, 2)

# simulation
tf = 5.0
dt = 0.1

obj = iLQR.Objective(Q,R,Qf,tf,x0,xf)
solver = iLQR.Solver(dp,obj,dt=dt);
U = 5.0*ones(solver.model.m,solver.N)

# Normal Solver
X_dp, U_dp = @time iLQR.solve(solver,U);

# Square Root Solver
solver.opts.square_root = true
X_sr, U_sr = @time iLQR.solve(solver,U);

plot(X_dp', label="full", color="red", width=3)
plot!(X_sr', label="square root",color="black")
