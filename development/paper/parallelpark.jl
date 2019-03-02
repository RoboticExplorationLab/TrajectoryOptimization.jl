using Plots
using Colors
pyplot()
include("N_plots.jl")

# Model and Objective
N = 51
model, obj = Dynamics.dubinscar_parallelpark
obj_uncon = UnconstrainedObjective(obj)

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-4
opts.resolve_feasible = false
opts.outer_loop_update_type = :default
opts.penalty_scaling = 200
opts.penalty_initial = .1
opts.R_infeasible = 20
dircol_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)

#### UNCONSTRAINED #####
solver = Solver(model, obj_uncon, N=N, opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(m,N)
X0 = line_trajectory(solver)
X0_rollout = rollout(solver,U0)

res_i, stats_i = solve(solver,U0)
stats_i["iterations"]
stats_i["runtime"]
evals(solver,:f) / stats_i["iterations"]

res_p, stats_p = solve_dircol(solver,X0_rollout,U0)
stats_p["iterations"]
evals(solver,:f)/stats_p["iterations"]
stats_p["runtime"]

Ns = [101,201,301,401]
group = "parallelpark/unconstrained"
run_step_size_comparison(model, obj_uncon, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=true)
plot_stat("runtime",group,legend=:bottomright,["rk3","ipopt"],title="Unconstrained Parallel Park")
plot_stat("iterations",group,legend=:bottom,["rk3","ipopt"],title="Unconstrained Parallel Park")
plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Unconstrained Parallel Park")
plot_stat("std",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Unconstrained Parallel Park")

#### CONSTRAINED ####
solver = Solver(model, obj, N=N, opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(m,N)
X0 = line_trajectory(solver)
X0_rollout = rollout(solver,U0)

res_i, stats_i = solve(solver,U0)
res_p, stats_p = solve_dircol(solver,X0_rollout,U0)

stats_i["iterations"]
stats_i["runtime"]
evals(solver,:f) / stats_i["iterations"]

stats_p["iterations"]
evals(solver,:f)/stats_p["iterations"]
stats_p["runtime"]

Ns = [101,201,301,401]
group = "parallelpark/constrained"
run_step_size_comparison(model, obj, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=true)
plot_stat("runtime",group,legend=:bottomright,["rk3","ipopt",""],title="Constrained Parallel Park",color=[:blue :darkorange2])
savefig(joinpath(IMAGE_DIR,"ppark_runtime.eps"))
plot_stat("iterations",group,legend=:bottom,["rk3","ipopt"],title="Constrained Parallel Park",color=[:blue :darkorange2])
savefig(joinpath(IMAGE_DIR,"ppark_iterations.eps"))
plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Constrained Parallel Park")
Plots.eps(joinpath(IMAGE_DIR,"ppark_runtime"))

# Constraint vs time (Newton tail)
# using TrajectoryOptimization: gen_usrfun_newton, NewtonVars, gen_newton_functions, newton_projection
# t_start = time_ns()
# V_ = newton_projection(solver,res_i,eps=1e-8,verbose=false)
# res_ = ConstrainedVectorResults(solver,V_.Z.X,V_.Z.U)
# backwardpass!(res_,solver)
# rollout!(res_,solver,0.0)
# max_violation(res_)
# t_newton = float(time_ns()-t_start)/1e9
#
# t_i = 0.469548  # from running @btime
# t_p = 0.659831  # from running @btime
# t_2 = 11.325    # from running @btime
# time_i = collect(range(0,stop=t_i,length=stats_i["iterations"]))
# time_p = range(0,stop=t_p,length=stats_p["iterations"])
# p = plot(time_p,stats_p["c_max"][2:end],yscale=:log10,label="Ipopt",color=:blue,width=2,
#     markershape=:circle,markerstrokecolor=:blue)
# c_max = [stats_i["c_max"]; max_violation(res_)]
# push!(time_i,t_i+t_newton)
# plot!(time_i,c_max,label="ALTRO*",color=:green,width=2,markershape=:circle,markerstrokecolor=:green,
#     xlabel="runtime (s)",ylabel="max constraint violation")
# plot_vertical_lines!(p,[t_i])
# time_2 = range(0,step=time_i[2]-time_i[1],length=stats_2["iterations"])
# plot!(time_2,stats_2["c_max"],label="ALTRO",width=2,color=:darkorange2,
#     markershape=:circle,markerstrokecolor=:darkorange2,xlim=[0,1.5],size=(500,250))
# savefig(p,joinpath(IMAGE_DIR,"ppark_newton.eps"))

# Constraint vs Iteration (Newton tail)
using TrajectoryOptimization: gen_usrfun_newton, NewtonVars, gen_newton_functions, newton_projection
t_start = time_ns()
solver.opts.iterations = 25
solver.opts.constraint_tolerance = 1e-2
res_i, stats_i = solve(solver,U0)
max_violation(res_i)

plot(stats_i["c_max"],yscale=:log10)

V_ = newton_projection(solver,res_i,eps=1e-8,verbose=false)
res_ = ConstrainedVectorResults(solver,V_.Z.X,V_.Z.U)
backwardpass!(res_,solver)
rollout!(res_,solver,0.0)
max_violation(res_)
t_newton = float(time_ns()-t_start)/1e9

t_i = 0.469548  # from running @btime
t_p = 0.659831  # from running @btime
t_2 = 11.325    # from running @btime
time_i = collect(range(0,stop=t_i,length=stats_i["iterations"]))
time_p = range(0,stop=t_p,length=stats_p["iterations"])
p = plot(time_p,stats_p["c_max"][2:end],yscale=:log10,label="Ipopt",color=:blue,width=2,
    markershape=:circle,markerstrokecolor=:blue)
c_max = [stats_i["c_max"]; max_violation(res_)]
push!(time_i,t_i+t_newton)
plot!(time_i,c_max,label="ALTRO*",color=:green,width=2,markershape=:circle,markerstrokecolor=:green,
    xlabel="runtime (s)",ylabel="max constraint violation")
plot_vertical_lines!(p,[t_i])
time_2 = range(0,step=time_i[2]-time_i[1],length=stats_2["iterations"])
plot!(time_2,stats_2["c_max"],label="ALTRO",width=2,color=:darkorange2,
    markershape=:circle,markerstrokecolor=:darkorange2,xlim=[0,1.5],size=(500,250))
savefig(p,joinpath(IMAGE_DIR,"ppark_newton.eps"))

# #### INFEASIBLE ####
# opts = SolverOptions()
# opts.verbose = false
# opts.cost_tolerance = 1e-6
# opts.cost_tolerance_intermediate = 1e-3
# opts.cost_tolerance_infeasible = 1e-4
# opts.constraint_tolerance = 1e-4
# opts.resolve_feasible = false
# opts.outer_loop_update_type = :default
# opts.use_nesterov = true
# opts.penalty_scaling = 200
# opts.penalty_initial = 10
# opts.R_infeasible = 10
# opts.square_root = true
# opts.constraint_decrease_ratio = 0.25
# opts.penalty_update_frequency = 2
#
# solver = Solver(model, obj, N=101, opts=opts)
# n,m,N = get_sizes(solver)
# U0 = ones(m,N)
# X0 = line_trajectory(solver)
# X0_rollout = rollout(solver,U0)
#
# solver.opts.verbose = false
# solver.opts.resolve_feasible = true
# solver.opts.cost_tolerance_infeasible = 1e-5
# @time res_i, stats_i = solve(solver,X0,U0)
# stats_i["iterations"]
# res_i.U[1]
# stats_i["runtime"]
# stats_i["iterations (infeasible)"]
# @time res_p, stats_p = solve_dircol(solver,X0_rollout,U0)
# # res_s, stats_s = solve_dircol(solver,X0_rollout,U0,nlp=:snopt)
#
# # import TrajectoryOptimization: _solve, get_feasible_trajectory
# # results = copy(res_i)
# # solver.state.infeasible = true
# # results_feasible = get_feasible_trajectory(results, solver)
# # results_feasible.λ_prev[1]
#
# # solver.state.infeasible
# # res, stats = _solve(solver,to_array(results_feasible.U),prevResults=results_feasible); stats["iterations"]
# # res, stats = _solve(solver,to_array(results_feasible.U)); stats["iterations"]
# # res, stats = _solve(solver,to_array(results_feasible.U),λ=results_feasible.λ); stats["iterations"]
# # res, stats = _solve(solver,to_array(results_feasible.U),λ=results_feasible.λ,μ=results_feasible.μ); stats["iterations"]
# #
#
# constraint_plot(solver,X0,U0)
#
# Ns = [101,201,301,401]
# group = "parallelpark/infeasible"
# run_step_size_comparison(model, obj, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=true, infeasible=true)
# plot_stat("runtime",group,legend=:bottomright,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)",ylim=[0,1.5])
# plot_stat("iterations",group,legend=:bottom,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)")
# plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Constrained Parallel Park (infeasible)")
#
#
#
# Combined Plot
group = "parallelpark/constrained"
Ns, data = load_data("runtime","ipopt","parallelpark/constrained")
Ns, err = load_data("std","ipopt","parallelpark/constrained")
p1 = plot(Ns,data,yerr=err,label="DIRCOL",color=:blue,marker=:circle,markerstrokecolor=:blue,ylabel="runtime",markersize=6)
Ns, data = load_data("runtime","rk3","parallelpark/constrained")
Ns, err = load_data("std","rk3","parallelpark/constrained")
plot!(Ns,data,yerr=err,label="ALTRO",color=:darkorange2,marker=:circle,markerstrokecolor=:darkorange2,markersize=6,ylim=(0,1.9))
Ns, data = load_data("runtime","rk3","parallelpark/infeasible")
Ns, err = load_data("std","rk3","parallelpark/infeasible")
plot!(Ns,data,yerr=err,label="ALTRO (inf)",color=:darkorange2,style=:dash,
    marker=:utriangle,markerstrokecolor=:darkorange2,markersize=8,
    title="Constrained",titlefontsize=10)

Ns, err = load_data("std","rk3","parallelpark/infeasible")

Ns, data = load_data("runtime",["rk3","ipopt"],"parallelpark/unconstrained")
Ns, err = load_data("std",["rk3","ipopt"],"parallelpark/unconstrained")
p2 = plot(Ns,data[2],yerr=err[2],color=:blue,style=:dot,label="DIRCOL", width=1.5,
    marker=:square,markerstrokecolor=:blue,legend=:topleft,markersize=4,ylim=ylims(p1))
plot!(Ns,data[1],yerr=err[1],color=:darkorange2,style=:dot,label="ALTRO",width=1.5,
    marker=:square,markerstrokecolor=:darkorange2,markersize=4,
    title="Unconstrained",titlefontsize=10)
plot(p1,p2,layout=(1,2),size=(500,300),xlabel="Knot points",ylabel="Runtime")
savefig(joinpath(IMAGE_DIR,"ppark_runtime.tiff"))
