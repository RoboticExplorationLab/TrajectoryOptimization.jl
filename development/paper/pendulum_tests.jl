
# Read in the system
model,obj = Dynamics.pendulum!
obj_c = Dynamics.pendulum_constrained![2]
obj_c.u_min[1] = -2
obj_c.u_max[1] = 2

# Params
N = 51
method = :hermite_simpson

# Initial Trajectory
U0 = ones(1,N)
X0 = line_trajectory(obj.x0,obj.xf,N)

X0_rollout = copy(X0)
solver = Solver(model,obj_c,N=N)
rollout!(X0_rollout,U0,solver)

# Knot point test params
Ns = [6,11,15,21,41,51,81,101,201,401,501,801,1001]
dt_truth = 1e-3

obj.tf ./ (Ns.-1)
N_truth = Int(obj.tf / dt_truth + 1)


#####################################
#          UNCONSTRAINED            #
#####################################
# Solver Options
opts = SolverOptions()
opts.use_static = false
opts.cost_tolerance = 1e-4
opts.outer_loop_update_type = :default
opts.iterations = 500

group  = "pendulum/unconstrained"
solver_truth, res_truth,  = run_dircol_truth(model, obj, dt_truth, X0_rollout, U0, group)
time_truth = get_time(solver_truth)
plot(time_truth, res_truth.X')

err_mid, eterm_mid, stats_mid = run_Ns(model, obj, Ns, :midpoint)
err_rk3, eterm_rk3, stats_rk3 = run_Ns(model, obj, Ns, :rk3)
err_foh, eterm_foh, stats_foh = run_Ns(model, obj, Ns, :rk3_foh)
err_rk4, eterm_rk4, stats_rk4 = run_Ns(model, obj, Ns, :rk4)

save_data(group)

plot_stat("error",group)
plot_stat("error_final",group,yscale=:log10)
plot_stat("runtime",group)
plot_stat("iterations",group)


#####################################
#            CONSTRAINED            #
#####################################
opts = SolverOptions()
opts.cost_tolerance = 1e-4
opts.cost_tolerance_intermediate = 1e-4
opts.constraint_tolerance = 1e-3
opts.outer_loop_update_type = :individual
opts.use_static = false
opts.constraint_decrease_ratio = .25

group = "pendulum/constrained"
solver_truth, res_truth,  = run_dircol_truth(model, obj_c, dt_truth, X0_rollout, U0, group)
time_truth = get_time(solver_truth)
plot(res_truth.X')

Ns = [21,41,51,81,101,201,401,501,801,1001]
err_mid, eterm_mid, stats_mid = run_Ns(model, obj_c, Ns, :midpoint)
err_rk3, eterm_rk3, stats_rk3 = run_Ns(model, obj_c, Ns, :rk3)
err_foh, eterm_foh, stats_foh = run_Ns(model, obj_c, Ns, :rk3_foh)
err_rk4, eterm_rk4, stats_rk4 = run_Ns(model, obj_c, Ns, :rk4)

save_data(group)

plot_stat("error",group)
plot_stat("error_final",group,yscale=:log10)
plot_stat("runtime",group,legend=:bottomright)
plot_stat("iterations",group)
plot_stat("c_max",group,yscale=:log10)


opts = SolverOptions()
opts.cost_tolerance = 1e-16
opts.cost_tolerance_intermediate = 1e-16
opts.constraint_tolerance = 1e-16
opts.use_static = false
opts.iterations = 250
opts.iterations_outerloop = 10

solver = solver(model, obj_c, opts=opts, N=51, integration=:rk3)


#####################################
#            INFEASIBLE             #
#####################################
# Solver Options
opts = SolverOptions()
opts.cost_tolerance = 1e-4
opts.cost_tolerance_intermediate = 1e-1
opts.constraint_tolerance = 1e-3
opts.outer_loop_update_type = :individual
opts.constraint_decrease_ratio = .25
opts.use_static = false
opts.resolve_feasible = false

solver_truth, res_truth,  = run_dircol_truth(model, obj_c, dt_truth, X0, U0, "pendulum/infeasible")
time_truth = get_time(solver_truth)
plot(res_truth.X')

obj_c

Ns = [21,51,81,101,201,401,501,801,1001]
err_mid, eterm_mid, stats_mid = run_Ns(model, obj_c, Ns, :midpoint, infeasible=true)
err_rk3, eterm_rk3, stats_rk3 = run_Ns(model, obj_c, Ns, :rk3, infeasible=true)
err_foh, eterm_foh, stats_foh = run_Ns(model, obj_c, Ns, :rk3_foh, infeasible=true)
err_rk4, eterm_rk4, stats_rk4 = run_Ns(model, obj_c, Ns, :rk4, infeasible=true)

save_data("pendulum/infeasible")

plot_stat("error",group)
plot_stat("error_final",group,yscale=:log10)
plot_stat("runtime",group,legend=:bottomright)
plot_stat("iterations",group)
plot_stat("c_max",group,yscale=:log10)

solver = Solver(model, obj_c, opts=opts, N=51, integration=:rk3_foh)
disable_logging(Logging.Debug)
solver.opts.verbose = true
solver.opts.live_plotting = true
solver.opts.cost_tolerance_intermediate = 1e-6
solve(solver,X0,U0)
