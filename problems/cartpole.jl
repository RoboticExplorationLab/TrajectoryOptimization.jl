# Cartpole
T = Float64
model = TrajectoryOptimization.Dynamics.cartpole_model
model_d = rk3(model)
n = model.n; m = model.m

# costs
Q = 1.0*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.; 0.; 0.]
xf = [0.; pi; 0.; 0.] # (ie, swing up)

N = 101
tf = 5.
dt = tf/(N-1)
U0 = [0.01*rand(m) for k = 1:N-1]
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)

cartpole_problem = TrajectoryOptimization.Problem(model_d, obj, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(cartpole_problem, U0)
