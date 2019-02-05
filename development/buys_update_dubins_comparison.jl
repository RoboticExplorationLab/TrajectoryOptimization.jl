using Test
using Plots

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = true
opts.cost_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-6
opts.square_root = true
opts.outer_loop_update_type = :default
opts.constraint_tolerance_second_order_dual_update = sqrt(opts.constraint_tolerance)
opts.use_second_order_dual_update = true
opts.penalty_max = 1e8
opts.iterations_outerloop = 25
opts.live_plotting = false
model,obj = TrajectoryOptimization.Dynamics.dubinscar_parallelpark

solver = Solver(model, obj, integration=:rk4, N=100, opts=opts)
U0 = rand(solver.model.m,solver.N-1)

results, stats = TrajectoryOptimization.solve(solver,U0)

@show stats["iterations"]
@show stats["outer loop iterations"]
@show stats["c_max"][end]
@show max_violation(results) < opts.constraint_tolerance

plot(stats["c_max"],yscale=:log10,ylabel="c_max",label="first")


plot(to_array(results.U)',label="")
plot(to_array(results.X)',label="")
