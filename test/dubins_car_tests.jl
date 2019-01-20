using Random
Random.seed!(7)

dt = 0.01
integration = :rk4

###################
## Parallel Park ##
###################

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5
opts.resolve_feasible = true
opts.outer_loop_update_type = :default
opts.R_infeasible = 10

# Set up model, objective, and solver
model, obj = TrajectoryOptimization.Dynamics.dubinscar_parallelpark
solver = Solver(model, obj, integration=integration, dt=dt, opts=opts)
U0 = rand(solver.model.m,solver.N-1)
X0 = line_trajectory(solver)

results, stats = TrajectoryOptimization.solve(solver,U0)
results_inf, stats_inf = TrajectoryOptimization.solve(solver,X0,U0)

@test norm(results.X[end]-obj.xf) < 1e-5
@test TrajectoryOptimization.max_violation(results) < 1e-5
@test norm(results_inf.X[end]-obj.xf) < 1e-5
@test TrajectoryOptimization.max_violation(results_inf) < 1e-5

# # Parallel Park (boxed)
# x_min = obj.x_min
# x_max = obj.x_max
# plt = plot(title="Parallel Park")#,aspect_ratio=:equal)
# plot!(x_min[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
# plot!(x_max[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
# plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_min[2]*ones(1000),color=:red,width=2,label="")
# plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_max[2]*ones(1000),color=:red,width=2,label="")
# plot_trajectory!(to_array(results.X),width=2,color=:green,label="Constrained",legend=:bottomright)
# plot_trajectory!(to_array(results_inf.X),width=2,color=:yellow,label="Constrained (infeasible)",legend=:bottomright)

########################
## Obstacle Avoidance ##
########################

model, obj_con_obstacles, circles = TrajectoryOptimization.Dynamics.dubinscar_obstacles
model, obj_con_obstacles_control, = TrajectoryOptimization.Dynamics.dubinscar_obstacles_control_limits
solver_con_obstacles = Solver(model, obj_con_obstacles_control, integration=integration, dt=dt, opts=opts)

# -Initial state and control trajectories
U0 = ones(solver_con_obstacles.model.m,solver_con_obstacles.N-1)
X0 = line_trajectory(solver_con_obstacles)
results_inf, stats_inf = TrajectoryOptimization.solve(solver_con_obstacles,X0,U0)


# plt = plot(title="Obstacle Avoidance")
# plot_obstacles(circles)
# plot_trajectory!(to_array(results_con_obstacles.X),width=2,color=:orange,label="Infeasible")
# plot(to_array(results.U)')

@test norm(results_inf.X[end]-obj_con_obstacles.xf) < 1e-5
@test TrajectoryOptimization.max_violation(results_inf) < 1e-5

# TODO add escape
