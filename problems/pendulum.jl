# Pendulum
T = Float64
model = TrajectoryOptimization.Dynamics.pendulum_model
model_d = rk3(model)
n = model.n; m = model.m

# costs
Q = 1.0e-1*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)

N = 51
dt = 0.1
U0 = [rand(m) for k = 1:N-1]
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)

pendulum_problem = TrajectoryOptimization.Problem(model_d, obj, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(pendulum_problem, U0)
rollout!(pendulum_problem)
