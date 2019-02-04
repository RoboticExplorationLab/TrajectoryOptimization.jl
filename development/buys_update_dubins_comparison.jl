using Test
using Plots

U0 = rand(solver.model.m,solver.N-1)

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5
opts.square_root = true
opts.outer_loop_update_type = :default
opts.constraint_tolerance_second_order_dual_update = sqrt(opts.constraint_tolerance)
opts.use_second_order_dual_update = true
opts.penalty_max = 1e4
opts.iterations_outerloop = 25
opts.gradient_type = :AuLa
opts.live_plotting = true
model,obj = TrajectoryOptimization.Dynamics.dubinscar_parallelpark

x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,3)
Q = (0.0)*Diagonal(I,3)
R = (1e-2)*Diagonal(I,2)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

solver = Solver(model, obj, integration=:rk4, N=100, opts=opts)

results, stats = TrajectoryOptimization.solve(solver,U0)

@show stats["iterations"]
@show stats["outer loop iterations"]
@show stats["c_max"][end]
@show max_violation(results) < opts.constraint_tolerance

plot(stats["c_max"],ylabel="c_max",label="first")


plot(to_array(results.U)',label="")
plot(to_array(results.X)',label="")
