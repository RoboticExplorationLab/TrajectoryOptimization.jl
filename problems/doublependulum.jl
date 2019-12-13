# # Double Pendulum
# T = Float64
# model = TrajectoryOptimization.Dynamics.doublependulum
# model_d = rk3(model)
# n = model.n; m = model.m
#
# # costs
# Qf = 100.0*Diagonal(I,n)
# Q = (1.0e-2)*Diagonal(I,n)
# R = (1.0e-2)*Diagonal(I,m)
#
# x0 = [0.; 0.; 0.; 0.]
# xf = [π; 0.; 0.; 0.];
# N = 101
# dt = 0.01
# U0 = [ones(m) for k = 1:N-1]
# obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)
#
# doublependulum = TrajectoryOptimization.Problem(model_d, obj, x0=x0, xf=xf, N=N, dt=dt)
# initial_controls!(doublependulum, U0)
