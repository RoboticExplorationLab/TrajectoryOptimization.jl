# Double Integrator
T = Float64
model = TrajectoryOptimization.Dynamics.doubleintegrator_model
model_d = rk3(model)
n = model.n; m = model.m

# costs
Q = 1.0*Diagonal(I,n)
Qf = 0.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.]
xf = [1.; 0] # (ie, swing up)

N = 21
dt = 0.1
U0 = [0.001*rand(m) for k = 1:N-1]
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)

u_max = 3.
u_min = -3.

bnd = BoundConstraint(n,m,u_max=u_max, u_min=u_min,trim=true)
goal = goal_constraint(xf)
constraints = Constraints([bnd],N)

doubleintegrator_problem = TrajectoryOptimization.Problem(model_d, obj, constraints=constraints, x0=x0, xf=xf, N=N, dt=dt)
doubleintegrator_problem.constraints[N] += goal
initial_controls!(doubleintegrator_problem, U0)
rollout!(doubleintegrator_problem)
