using Random
Random.seed!(7)

dt = 0.01
integration = :rk4

###################
## Parallel Park ##
###################

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-8
opts.cost_tolerance_intermediate = 1e-8
opts.constraint_tolerance = 1e-8
opts.resolve_feasible = true
opts.outer_loop_update_type = :default
opts.R_infeasible = 10

# Set up model, objective, and solver
model, obj = TrajectoryOptimization.Dynamics.dubinscar_parallelpark
solver = Solver(model, obj, integration=integration, dt=dt, opts=opts)
U0 = ones(solver.model.m,solver.N-1)
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

############
## Escape ##
############

N_escape = 101
model, obj_escape, circles_escape = TrajectoryOptimization.Dynamics.dubinscar_escape
solver_escape = Solver(model, obj_escape, integration=integration, N=N_escape, opts=opts)
X_guess = [2.5 2.5 0.;4. 5. .785;5. 6.25 0.;7.5 6.25 -.261;9 5. -1.57;7.5 2.5 0.]
X0 = TrajectoryOptimization.interp_rows(solver_escape.N,solver_escape.obj.tf,Array(X_guess'))
U0 = ones(solver_escape.model.m,solver_escape.N-1)

solver_escape.opts.R_infeasible = 1e-1
solver_escape.opts.resolve_feasible = true
solver_escape.opts.cost_tolerance = 1e-6
solver_escape.opts.cost_tolerance_intermediate = 1e-3
solver_escape.opts.constraint_tolerance = 1e-5
solver_escape.opts.constraint_tolerance_intermediate = 0.01
solver_escape.opts.penalty_scaling = 100.0
solver_escape.opts.penalty_initial = 100.0
solver_escape.opts.outer_loop_update_type = :default
solver_escape.opts.iterations_outerloop = 20
solver_escape.opts.use_penalty_burnin = false
solver_escape.opts.verbose = false
solver_escape.opts.live_plotting = false
results_escape, stats_escape = solve(solver_escape,X0,U0)

# plt = plot(title="Escape",aspect_ratio=:equal)
# plot_obstacles(circles_escape)
# plot_trajectory!(to_array(results_escape.X),width=2,color=:purple,label="Infeasible",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
# plot!(X0[1,:],X0[2,:],label="Infeasible Initialization",width=1,color=:purple,linestyle=:dash)
# display(plt)
# plot(to_array(results_escape.U[1:solver_escape.N-1])',label="")

@test norm(results_escape.X[end]-solver_escape.obj.xf) < 1e-5
@test TrajectoryOptimization.max_violation(results_escape) < 1e-5
