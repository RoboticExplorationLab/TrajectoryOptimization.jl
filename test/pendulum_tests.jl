# model
T = Float64

verbose = false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-6)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-4)

int_schemes = [:midpoint, :rk3, :rk4, :rk3_implicit, :midpoint_implicit]
model = TrajectoryOptimization.Dynamics.pendulum_model
n = model.n; m = model.m
xf = Problems.pendulum_problem.xf
N = Problems.pendulum_problem.N

## Unconstrained
for is in int_schemes
    prob = update_problem(copy(Problems.pendulum_problem),model=discretize_model(model,is))
    solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
    TrajectoryOptimization.solve!(prob, solver_ilqr)
    @test norm(prob.X[N] - xf) < 1.0e-3
end

## Constrained
u_bound = 3.0
bnd = BoundConstraint(n, m, u_min=-u_bound, u_max=u_bound)
goal = goal_constraint(xf)

constraints = Constraints([bnd],N)

for is in int_schemes
    prob = update_problem(copy(Problems.pendulum_problem),model=discretize_model(model,is),constraints=copy(constraints))
    solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
    TrajectoryOptimization.solve!(prob, solver_al)
    @test norm(prob.X[N] - xf,Inf) < 1.0e-3
    @test TrajectoryOptimization.max_violation(prob) < opts_al.constraint_tolerance
end

for is in int_schemes
    prob = update_problem(copy(Problems.pendulum_problem),model=discretize_model(model,is),constraints=copy(constraints))
    prob.constraints[N] += goal
    solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
    TrajectoryOptimization.solve!(prob, solver_al)
    @test norm(prob.X[N] - xf) < opts_al.constraint_tolerance
    @test TrajectoryOptimization.max_violation(prob) < opts_al.constraint_tolerance
end

# Test undefined integration
@test_throws ArgumentError TrajectoryOptimization.Problem(model, Problems.pendulum_problem.obj, integration=:bogus, N=N)

# Test different solve methods
prob = copy(Problems.pendulum_problem)
prob = update_problem(prob, constraints=constraints)
solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
out = solve!(prob, solver_al)
@test out isa AugmentedLagrangianSolver

prob = copy(Problems.pendulum_problem)
prob = update_problem(prob, constraints=constraints)
out = solve!(prob, opts_al)
@test out isa AugmentedLagrangianSolver

prob = copy(Problems.pendulum_problem)
prob = update_problem(prob, constraints=constraints)
solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
out = solve(prob, solver_al)
@test out isa Tuple{Problem, AugmentedLagrangianSolver}
@test isnan(cost(prob))

out = solve(prob, opts_al)
@test out isa Tuple{Problem, AugmentedLagrangianSolver}
@test isnan(cost(prob))


# Test unconstrained
prob = copy(Problems.pendulum_problem)
solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
@test !is_constrained(prob)
out = solve!(prob, solver_al)
@test out isa AugmentedLagrangianSolver

prob = copy(Problems.pendulum_problem)
out = solve!(prob, opts_al)
@test out isa iLQRSolver
