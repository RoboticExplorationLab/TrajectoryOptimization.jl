# Pendulum
T = Float64
model = TrajectoryOptimization.Dynamics.pendulum_model
model_d = rk3(model)
n = model.n; m = model.m

# costs
Q = Array(1e-3*Diagonal(I,n))
R = Array(1e-3*Diagonal(I,m))
Qf = Q
x0 = zeros(n)
xf = [pi;0.0]

u_bnd = 3.
bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
goal_con = goal_constraint(xf)

N = 31
dt = 0.15
U0 = [ones(m) for k = 1:N-1]
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)
constraints = Constraints(N)
for k = 1:N-1
    constraints[k] += bnd
end
constraints[N] += goal_con

pendulum_problem = TrajectoryOptimization.Problem(model_d, obj, constraints=constraints, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(pendulum_problem, U0)
