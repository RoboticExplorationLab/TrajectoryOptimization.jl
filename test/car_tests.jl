Random.seed!(7)

# model
T = Float64
integration = :rk4
model = TrajectoryOptimization.Dynamics.car_model
n = model.n; m = model.m

# cost
Q = (1e-2)*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)
x0 = [0.;0.;0.]
xf = [0.;1.;0.]
dt = 0.01

costfun = TrajectoryOptimization.LQRCost(Q, R, Qf, xf)

verbose=false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-5)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-5)
opts_altro = TrajectoryOptimization.ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al)

N = 101
dt = 0.1
U0 = [ones(m) for k = 1:N-1]
X0 = TrajectoryOptimization.line_trajectory(x0,xf,N)

# Parallel Park
model_d = discretize_model(model,integration,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)
TrajectoryOptimization.initial_controls!(prob, U0)
TrajectoryOptimization.solve!(prob, opts_ilqr)
@test norm(prob.X[N] - xf) < 1e-3

# Infeasible parallel park
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)
TrajectoryOptimization.initial_controls!(prob, U0)
copyto!(prob.X,X0)
@test norm(prob.X[N] - xf) < 1e-3
