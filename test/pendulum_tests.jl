# model
T = Float64

verbose = false
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,iterations=50,penalty_scaling=10.0)
opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_minimum_time=15.0,dt_max=0.15,dt_min=1.0e-3)

int_schemes = [:midpoint, :rk3, :rk4, :rk3_implicit, :midpoint_implicit]
model = TrajectoryOptimization.Dynamics.pendulum_model
n = model.n; m = model.m
xf = copy(Problems.pendulum_problem.xf)
N = copy(Problems.pendulum_problem.N)

for is in int_schemes
    prob = update_problem(copy(Problems.pendulum_problem),model=discretize_model(model,is))
    TrajectoryOptimization.solve!(prob, opts_altro)
    @test TrajectoryOptimization.max_violation(prob) < opts_al.constraint_tolerance
end

# Test undefined integration
@test_throws ArgumentError TrajectoryOptimization.Problem(model, Problems.pendulum_problem.obj, integration=:bogus, N=N)

# Test different solve methods
prob = copy(Problems.pendulum_problem)
solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
out = solve!(prob, solver_al)
@test out isa AugmentedLagrangianSolver

prob = copy(Problems.pendulum_problem)
out = solve!(prob, opts_al)
@test out isa AugmentedLagrangianSolver

prob = copy(Problems.pendulum_problem)
solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
out = solve(prob, solver_al)
@test out isa Tuple{Problem, AugmentedLagrangianSolver}
@test isnan(cost(prob))

out = solve(prob, opts_al)
@test out isa Tuple{Problem, AugmentedLagrangianSolver}
@test isnan(cost(prob))
