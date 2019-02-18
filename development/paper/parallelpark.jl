using Plots
using Colors
pyplot()
include("N_plots.jl")

# Model and Objective
model, obj = Dynamics.dubinscar_parallelpark
obj_uncon = UnconstrainedObjective(obj)

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
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
# res_s, stats_s = solve_dircol(solver,X0_rollout,U0,nlp=:snopt)

Ns = [101,201,301,401]
group = "parallelpark/unconstrained"
run_step_size_comparison(model, obj_uncon, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=true)
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
opts.penalty_initial = .001
opts.square_root = true
opts.iterations_outerloop = 50
opts.use_nesterov = true

solver = Solver(model, obj, N=101, opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(m,N)
X0 = line_trajectory(solver)
X0_rollout = rollout(solver,U0)

# constraint_plot(solver,U0,false)

@time res_i, stats_i = solve(solver,U0)
@time res_p, stats_p = solve_dircol(solver,X0_rollout,U0)
# res_s, stats_s = solve_dircol(solver,X0_rollout,U0,nlp=:snopt,grads=:none)

Ns = [101,201,301,401]
group = "parallelpark/constrained"
run_step_size_comparison(model, obj, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=true)
plot_stat("runtime",group,legend=:bottomright,["rk3","ipopt",""],title="Constrained Parallel Park",color=[:blue :darkorange2])
savefig(joinpath(IMAGE_DIR,"ppark_runtime.eps"))
plot_stat("iterations",group,legend=:bottom,["rk3","ipopt"],title="Constrained Parallel Park",color=[:blue :darkorange2])
savefig(joinpath(IMAGE_DIR,"ppark_iterations.eps"))
plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Constrained Parallel Park")
Plots.eps(joinpath(IMAGE_DIR,"ppark_runtime"))

#### INFEASIBLE ####
opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-3
opts.cost_tolerance_infeasible = 1e-4
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

solver = Solver(model, obj, N=51, opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(m,N)
X0 = line_trajectory(solver)
X0_rollout = rollout(solver,U0)

solver.opts.verbose = false
solver.opts.resolve_feasible = true
solver.opts.cost_tolerance_infeasible = 1e-4
@time res_i, stats_i = solve(solver,X0,U0)
stats_i["iterations"]
res_i.U[1]
stats_i["runtime"]
stats_i["iterations (infeasible)"]
@time res_p, stats_p = solve_dircol(solver,X0_rollout,U0)
# res_s, stats_s = solve_dircol(solver,X0_rollout,U0,nlp=:snopt)

# import TrajectoryOptimization: _solve, get_feasible_trajectory
# results = copy(res_i)
# solver.state.infeasible = true
# results_feasible = get_feasible_trajectory(results, solver)
# results_feasible.λ_prev[1]

# solver.state.infeasible
# res, stats = _solve(solver,to_array(results_feasible.U),prevResults=results_feasible); stats["iterations"]
# res, stats = _solve(solver,to_array(results_feasible.U)); stats["iterations"]
# res, stats = _solve(solver,to_array(results_feasible.U),λ=results_feasible.λ); stats["iterations"]
# res, stats = _solve(solver,to_array(results_feasible.U),λ=results_feasible.λ,μ=results_feasible.μ); stats["iterations"]
#

constraint_plot(solver,X0,U0)

Ns = [101,201,301,401]
group = "parallelpark/infeasible"
run_step_size_comparison(model, obj, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=true, infeasible=true)
plot_stat("runtime",group,legend=:bottomright,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)",ylim=[0,1.5])
plot_stat("iterations",group,legend=:bottom,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)")
plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)")



# Combined Plot
group = "parallelpark/constrained"
plot_stat("runtime",group,legend=:bottomright,["rk3","ipopt",""],title="Constrained Parallel Park",color=[:blue :darkorange2],label=["" "constrained"])
Ns, data = load_data("runtime","rk3","parallelpark/infeasible")
plot!(Ns,data,label="infeasible",marker=:utriangle,color=:darkorange2,style=:dash)
Ns, data = load_data("runtime",["rk3","ipopt"],"parallelpark/unconstrained")
plot!(Ns,data[1],color=:darkorange2,style=:dot,label="unconstrained",marker=:square)
plot!(Ns,data[2],color=:blue,style=:dot,label="",marker=:square,size=(500,250),legend=:topleft)
savefig(joinpath(IMAGE_DIR,"ppark_runtime.eps"))
