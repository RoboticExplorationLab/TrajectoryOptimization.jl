model = TrajectoryOptimization.Dynamics.pendulum_model
n = model.n; m = model.m
model_d = rk3(model)

# cost
Q = Array(1e-3*Diagonal(I,n))
R = Array(1e-3*Diagonal(I,m))
Qf = Q
x0 = zeros(n)
xf = [pi;0.0]

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,iterations=50,penalty_scaling=10.0)
opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=15.0,dt_max=0.15,dt_min=1.0e-3)

# constraints
u_bnd = 3.
bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
goal_con = goal_constraint(xf)

# problem
N = 31
U = [ones(m) for k = 1:N-1]
constraints = Constraints(N)
for k = 1:N-1
    constraints[k] += bnd
end
constraints[N] += goal_con
obj = LQRObjective(Q,R,Qf,xf,N)

dt = 0.15
pendulum_problem = Problem(model_d,obj,U,constraints=constraints,dt=dt,x0=x0,xf=xf,N=N)
