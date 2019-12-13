# model = TrajectoryOptimization.Dynamics.pendulum
# n = model.n; m = model.m
# model_d = rk3(model)
#
# # cost
# Q = Array(1e-3*Diagonal(I,n))
# R = Array(1e-3*Diagonal(I,m))
# Qf = Q
# x0 = zeros(n)
# xf = [pi;0.0]
#
# # options
# verbose=false
# opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)
# opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,iterations=50,penalty_scaling=10.0)
# opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=15.0,dt_max=0.15,dt_min=1.0e-3)
#
# # constraints
# u_bnd = 3.
# bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
# goal_con = goal_constraint(xf)
#
# # problem
# N = 31
# U = [ones(m) for k = 1:N-1]
# constraints = Constraints(N)
# for k = 1:N-1
#     constraints[k] += bnd
# end
# constraints[N] += goal_con
# obj = LQRObjective(Q,R,Qf,xf,N)
#
# N = 31
# dt = 0.15
# pendulum = Problem(model_d,obj,U,constraints=constraints,dt=dt,x0=x0,xf=xf,N=N)

# Static Problem
model = Dynamics.Pendulum()
n,m = size(model)

# cost
Q = 1e-3*Diagonal(@SVector ones(n))
R = 1e-3*Diagonal(@SVector ones(m))
Qf = 1e-3*Diagonal(@SVector ones(n))
x0 = @SVector zeros(n)
xf = @SVector [pi, 0.0]  # i.e. swing up
obj = LQRObjective(Q,R,Qf,xf,N)

# constraints
u_bnd = 3.
bnd = StaticBoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
goal_con = GoalConstraint(n,m,xf)

con_bnd = ConstraintVals(bnd, 1:N-1)
con_xf = ConstraintVals(goal_con, N:N)
conSet = ConstraintSets(n,m,[con_bnd, con_xf],N)

# problem
U = [@SVector ones(m) for k = 1:N-1]
pendulum_static = StaticProblem(model, obj, xf, tf, constraints=conSet, x0=x0)
initial_controls!(pendulum_static, U)
