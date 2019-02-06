using Plots
include("N_plots.jl")

# Model and Objective
model, obj = Dynamics.dubinscar_parallelpark
obj_uncon = UnconstrainedObjective(obj)

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5
opts.resolve_feasible = false
opts.outer_loop_update_type = :default
opts.use_nesterov = true
opts.penalty_scaling = 200
opts.penalty_initial = .1
opts.R_infeasible = 20

dircol_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)

# Params
N = 51

#### UNCONSTRAINED #####
solver = Solver(model, obj_uncon, N=N, opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(m,N)
X0 = line_trajectory(solver)
X0_rollout = rollout(solver,U0)

res_i, stats_i = solve(solver,U0)
res_p, stats_p = solve_dircol(solver,X0_rollout,U0)
res_s, stats_s = solve_dircol(solver,X0_rollout,U0,nlp=:snopt)

Ns = [51,101,201,301,401]
group = "parallelpark/unconstrained"
run_step_size_comparison(model, obj_uncon, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=false)
plot_stat("runtime",group,legend=:bottomright,["rk3","ipopt"],title="Unconstrained Parallel Park")
plot_stat("iterations",group,legend=:bottom,["rk3","ipopt"],title="Unconstrained Parallel Park")
plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Unconstrained Parallel Park")



#### CONSTRAINED ####
opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-2
opts.constraint_tolerance = 1e-5
opts.outer_loop_update_type = :default
opts.use_nesterov = true
opts.penalty_scaling = 100
opts.penalty_initial = .01
opts.square_root = true
opts.iterations_outerloop = 50

solver = Solver(model, obj, N=101, opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(m,N)
X0 = line_trajectory(solver)
X0_rollout = rollout(solver,U0)


@time res_i, stats_i = solve(solver,U0)
@time res_p, stats_p = solve_dircol(solver,X0_rollout,U0)
# res_s, stats_s = solve_dircol(solver,X0_rollout,U0,nlp=:snopt)

constraint_plot(solver,U0,title="Constrained Parallel Park")

Ns = [51,101,201,301]
group = "parallelpark/constrained"
run_step_size_comparison(model, obj, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt,:snopt],benchmark=false)
plot_stat("iterations",group,legend=:bottom,["rk3","ipopt","snopt"],title="Constrained Parallel Park")
plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt","snopt"],title="Constrained Parallel Park")



#### INFEASIBLE ####
solver = Solver(model, obj, N=51, opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(m,N)
X0 = line_trajectory(solver)
X0_rollout = rollout(solver,U0)


@time res_i, stats_i = solve(solver,X0,U0)
@time res_p, stats_p = solve_dircol(solver,X0_rollout,U0)
# res_s, stats_s = solve_dircol(solver,X0_rollout,U0,nlp=:snopt)

constraint_plot(solver,X0,U0)

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-3
opts.constraint_tolerance = 1e-5
opts.resolve_feasible = false
opts.outer_loop_update_type = :default
opts.use_nesterov = true
opts.penalty_scaling = 200
opts.penalty_initial = 10
opts.R_infeasible = 10
opts.square_root = true
opts.constraint_decrease_ratio = 0.25
opts.penalty_update_frequency = 2

Ns = [51,101,201,301]
group = "parallelpark/infeasible"
run_step_size_comparison(model, obj, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=false, infeasible=true)
plot_stat("runtime",group,legend=:bottomright,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)",ylim=[0,1.5])
plot_stat("iterations",group,legend=:bottom,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)")
plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)")
