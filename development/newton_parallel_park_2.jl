using Random
Random.seed!(7)

dt = 0.01
integration = :rk4

###################
## Parallel Park ##
###################

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5
opts.resolve_feasible = true
opts.outer_loop_update_type = :default
opts.R_infeasible = 10

# Set up model, objective, and solver
model, obj = TrajectoryOptimization.Dynamics.dubinscar_parallelpark
solver = Solver(model, obj, integration=integration, N=101, opts=opts)
U0 = ones(solver.model.m,solver.N-1)
X0 = line_trajectory(solver)

results, stats = TrajectoryOptimization.solve(solver,U0)
# results_inf, stats_inf = TrajectoryOptimization.solve(solver,X0,U0)

max_violation(results)
results_new = copy(results)

# newton_solve!(results_new,solver)

newton_results = NewtonResults(solver)
newton_active_set!(newton_results,results_new,solver)
# sum(newton_results.active_set)
# sum(vcat(results_new.active_set...))
# sum(newton_results.active_set_ineq)
# newton_results.s[findall(x->x != 0.0, newton_results.s)]
# findall(x->x < 0.0, vcat(results.C...)[newton_results.active_set_ineq])
update_newton_results!(newton_results,results_new,solver)
newton_step!(results_new,newton_results,solver,1.0,0.00000001)
max_violation(results_new)
