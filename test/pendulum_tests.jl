# model
T = Float64
model = TrajectoryOptimization.Dynamics.pendulum_model
n = model.n; m = model.m

# costs
Q = 1.0e-1*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-5)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-5)

N = 51
dt = 0.1
U0 = [rand(m) for k = 1:N-1]
int_schemes = [:midpoint, :rk3, :rk4]

## Unconstrained
for is in int_schemes
    prob = TrajectoryOptimization.Problem(model, TrajectoryOptimization.ObjectiveNew(costfun,N), integration=is, x0=x0, N=N, dt=dt)
    TrajectoryOptimization.initial_controls!(prob, U0)
    solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
    TrajectoryOptimization.solve!(prob, solver_ilqr)
    @test norm(prob.X[N] - xf) < 1e-4
end

## Constrained
u_bound = 3.0
bnd = TrajectoryOptimization.bound_constraint(n, m, u_min=-u_bound, u_max=u_bound)
con = [bnd]

for is in int_schemes
    prob = TrajectoryOptimization.Problem(model, TrajectoryOptimization.ObjectiveNew(costfun,N),
        constraints=TrajectoryOptimization.ProblemConstraints(con,N),integration=is, x0=x0, N=N, dt=dt)
    TrajectoryOptimization.initial_controls!(prob, U0)
    solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
    TrajectoryOptimization.solve!(prob, solver_al)
    @test norm(prob.X[N] - xf) < 1e-4
    @test TrajectoryOptimization.max_violation(prob) < opts_al.constraint_tolerance
end

goal = TrajectoryOptimization.goal_constraint(xf)
con = [bnd,goal]

for is in int_schemes
    prob = TrajectoryOptimization.Problem(model, TrajectoryOptimization.ObjectiveNew(costfun,N),
        constraints=TrajectoryOptimization.ProblemConstraints(con,N),integration=is, x0=x0, N=N, dt=dt)
    TrajectoryOptimization.initial_controls!(prob, U0)
    solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
    TrajectoryOptimization.solve!(prob, solver_al)
    @test norm(prob.X[N] - xf) < opts_al.constraint_tolerance
    @test TrajectoryOptimization.max_violation(prob) < opts_al.constraint_tolerance
end

# Test undefined integration
@test_throws ArgumentError TrajectoryOptimization.Problem(model, TrajectoryOptimization.ObjectiveNew(costfun,N), integration=:bogus, N=N)
