using Test
using Plots
U = rand(solver.model.m, solver.N)

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = true
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-6
opts.constraint_tolerance = 1e-6
opts.square_root = true
opts.outer_loop_update_type = :default
opts.constraint_tolerance_second_order_dual_update = sqrt(opts.constraint_tolerance)
opts.use_second_order_dual_update = true
opts.penalty_max = 1e8
opts.iterations_outerloop = 25
opts.live_plotting = true
model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -.15
u_max = 0.2
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=5.0, u_min=u_min, u_max=u_max)#, x_min=x_min, x_max=x_max)

integrator = :rk4
dt = 0.05
solver = TrajectoryOptimization.Solver(model,obj_con,integration=integrator,dt=dt,opts=opts)
results, stats = TrajectoryOptimization.solve(solver,U)

@show stats["iterations"]
@show stats["outer loop iterations"]
@show stats["c_max"][end]
@show max_violation(results) < opts.constraint_tolerance

plot!(stats["c_max"],ylabel="c_max",label="buys")


plot(to_array(results.U)',label="")
plot(to_array(results.X)',label="")
