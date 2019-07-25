# model
T = Float64

verbose = false
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,iterations=50,penalty_scaling=10.0)
opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_minimum_time=15.0,dt_max=0.15,dt_min=1.0e-3)

int_schemes = [:midpoint, :rk3, :rk4, :rk3_implicit, :midpoint_implicit]
model = TrajectoryOptimization.Dynamics.pendulum
n = model.n; m = model.m
xf = Problems.pendulum.xf
N = Problems.pendulum.N

# Make sure Ipopt will solve a discrete problem by converting it
prob = copy(Problems.pendulum)
opts_ipopt = DIRCOLSolverOptions(verbose=false)
res, = solve(prob, opts_ipopt)
@test prob isa Problem{T,Discrete}
@test res isa Problem{T,Continuous}


for is in int_schemes
    prob = update_problem(copy(Problems.pendulum),model=discretize_model(model,is))
    TrajectoryOptimization.solve!(prob, opts_altro)
    @test TrajectoryOptimization.max_violation(prob) < opts_al.constraint_tolerance
end

# Test undefined integration
@test_throws ArgumentError TrajectoryOptimization.Problem(model, Problems.pendulum.obj, integration=:bogus, N=N)

# Test different solve methods
prob = copy(Problems.pendulum)
solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
out = solve!(prob, solver_al)
@test out isa AugmentedLagrangianSolver

prob = copy(Problems.pendulum)
out = solve!(prob, opts_al)
@test out isa AugmentedLagrangianSolver

prob = copy(Problems.pendulum)
solver_al = TrajectoryOptimization.AugmentedLagrangianSolver(prob, opts_al)
out = solve(prob, solver_al)
@test out isa Tuple{Problem, AugmentedLagrangianSolver}
@test isnan(cost(prob))

out = solve(prob, opts_al)
@test out isa Tuple{Problem, AugmentedLagrangianSolver}
@test isnan(cost(prob))
