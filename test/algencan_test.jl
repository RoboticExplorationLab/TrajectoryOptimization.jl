using Random
using Logging, Plots
Random.seed!(7)

dt = 0.01
integration = :rk4

###################
## Parallel Park ##
###################

opts = SolverOptions()
opts.verbose = true
opts.cost_tolerance = 1e-7
opts.cost_tolerance_intermediate = 1e-6
opts.constraint_tolerance = 1e-7
opts.resolve_feasible = true
opts.outer_loop_update_type = :default
opts.R_infeasible = 10
opts.square_root = true

# Set up model, objective, and solver
model, obj = TrajectoryOptimization.Dynamics.dubinscar_parallelpark
solver = Solver(model, obj, integration=integration, dt=dt, opts=opts)
U0 = ones(solver.model.m,solver.N-1)
X0 = line_trajectory(solver)

res, stats = TrajectoryOptimization.solve(solver,U0)

solver2 = Solver(solver)
solver2.opts.al_type = :algencan
res2, stats2 = solve(solver2,U0)

plot()
plot_trajectory!(res2)

p = plot(stats["c_max"],yscale=:log10)
plot!(stats2["c_max"])
plot_vertical_lines!(p,stats2["outer_updates"])
stats2["major iterations"]
stats2
