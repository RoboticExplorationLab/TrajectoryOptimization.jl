# model
T = Float64
model = TrajectoryOptimization.Dynamics.pendulum_frictionless_model
n = model.n; m = model.m

# costs
Q = 1.0e-1*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-6)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-4)

N = 201
tf = 1.0
dt = tf/(N-1)
U0 = [rand(m) for k = 1:N-1]


int_exp = [:midpoint, :rk3, :rk4]

## Unconstrained
for is in int_exp
    model_d = discretize_model(model,is,dt)
    prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)
    TrajectoryOptimization.initial_controls!(prob, U0)
    solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
    TrajectoryOptimization.solve!(prob, solver_ilqr)
    @test norm(prob.X[N] - xf) < 1.0e-3
end

int_imp = [:midpoint_implicit, :rk3_implicit]

for is in int_imp
    model_d = discretize_model(model,is,dt)
    prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)
    TrajectoryOptimization.initial_controls!(prob, U0)
    solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
    TrajectoryOptimization.solve!(prob, solver_ilqr)
    @test norm(prob.X[N] - xf) < 1.0e-3

    Xf = copy(prob.X[end])
    rollout_reverse!(prob,Xf)
    @test norm(prob.X[1] - x0) < 1.0e-3
end
