using Plots
using Random
pyplot()
include("N_plots.jl")
model, obj, circles = Dynamics.dubinscar_escape
Random.seed!(2);
N=101
opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 0.01
opts.constraint_tolerance = 1e-4
opts.resolve_feasible = true
opts.outer_loop_update_type = :default
opts.penalty_scaling = 50
opts.penalty_initial = 10
opts.R_infeasible = 1
opts.square_root = true
ipopt_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)

# Constrained
solver = Solver(model,obj,N=N,opts=opts)
n,m,N = get_sizes(solver)

U0 = ones(m,N-1) + rand(m,N-1)/3
X_guess = [2.5 2.5 0.;4. 5. .785;5. 6.25 0.;7.5 6.25 -.261;9 5. -1.57;7.5 2.5 0.]
X0 = TrajectoryOptimization.interp_rows(N,obj.tf,Array(X_guess'))
# plot(X0[1,:],X0[2,:])

res, stats = solve(solver,U0)
res_inf, stats_inf = solve(solver,X0,U0)
@btime solve($solver,$X0,$U0)
stats_inf["iterations"]
stats_inf["runtime"]
evals(solver,:f) / stats_inf["iterations"]

res_i, stats_i = solve_dircol(solver,X0,U0, options=ipopt_options)
@btime solve_dircol($solver,$X0,$U0, options=$ipopt_options)
stats_i["iterations"]
evals(solver,:f)/stats_i["iterations"]
stats_i["runtime"]

plt = plot(title="",aspect_ratio=:equal,size=(500,300),xlim=[-0.5,10.5],ylim=[-0.5,6.6],grid=:off,bg=:white)
plot_obstacles(circles,:grey)
scatter!(X_guess[:,1],X_guess[:,2],marker=:circle,markersize=7,markerstrokecolor=:gold,color=:gold,label="")
scatter!((X_guess[end,1],X_guess[end,2]),marker=:circle,markersize=7,markerstrokecolor=:gold,color=:gold,label="Initial Guess")
plot_trajectory!(X0,style=:dot,color=:black,width=2,label="Interpolation")
plot_trajectory!(res_i.X,width=3,color=:blue,label="DIRCOL")
plot_trajectory!(to_array(res.X),width=2,color=:darkorange2,label="ALTRO",aspect_ratio=:equal,legend=:topleft)
plot_trajectory!(to_array(res_inf.X),width=2,color=:darkorange2,style=:dash,label="ALTRO (inf)",aspect_ratio=:equal)
scatter!([obj.x0[1]],[obj.x0[2]],label=:start,color=:red,markerstrokecolor=:red,markersize=12,markershape=:rtriangle)
scatter!([obj.xf[1]],[obj.xf[2]],label=:goal,color=:green,markerstrokecolor=:green,markersize=12,markershape=:rtriangle,axis=:off)

savefig(joinpath(IMAGE_DIR,"escape_traj.svg"))

# Newton solve comparison
Random.seed!(2);
opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 0.01
opts.constraint_tolerance = 1e-8
opts.resolve_feasible = true
opts.outer_loop_update_type = :default
opts.penalty_scaling = 50
opts.penalty_initial = 10
opts.R_infeasible = 1
opts.square_root = true
ipopt_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)

solver = Solver(model,obj,N=N,opts=opts)
n,m,N = get_sizes(solver)

U0 = ones(m,N-1) + rand(m,N-1)/3
X_guess = [2.5 2.5 0.;4. 5. .785;5. 6.25 0.;7.5 6.25 -.261;9 5. -1.57;7.5 2.5 0.]
X0 = TrajectoryOptimization.interp_rows(N,obj.tf,Array(X_guess'))

res_inf, stats_inf = solve(solver,X0,U0)
stats_inf["iterations"]
stats_inf["runtime"]
evals(solver,:f) / stats_inf["iterations"]

res_i, stats_i = solve_dircol(solver,X0,U0, options=ipopt_options)
stats_i["iterations"]
evals(solver,:f)/stats_i["iterations"]
stats_i["runtime"]

# Newton results
solver.opts.resolve_feasible = false
solver.opts.feasible_projection = false
idx_newton = 13
solver.opts.iterations = idx_newton # cut out at this constraint violation
@btime solve(solver,$X0,$U0)
res_n, stats_n = solve(solver,X0,U0)
max_violation(res_n)
stats_n["c_max"][end]

length(stats_n["c_max"])

using TrajectoryOptimization: gen_usrfun_newton, NewtonVars, gen_newton_functions, newton_projection
t_start = time_ns()
V_ = newton_projection(solver,res_n,eps=1e-12,verbose=false)
res_ = ConstrainedVectorResults(solver,V_.Z.X,V_.Z.U)
backwardpass!(res_,solver)
rollout!(res_,solver,0.0)
# max_violation(res_)
t_newton = float(time_ns()-t_start)/1e9

t_ilqr = 0.617336
t_dircol = 9.08
t_inf = stats_inf["runtime"]
time_newton = collect(range(0,t_ilqr,length=stats_n["iterations"]))
push!(time_newton,t_ilqr + t_newton)
time_dircol = range(0,t_dircol,length=stats_i["iterations"])
time_ilqr = range(0,length=stats_inf["iterations"],step=time_newton[2]-time_newton[1])
c_max = [stats_n["c_max"]; max_violation(res_)]
stats_inf["c_max"][31]

markersize = 3
width = 1.5
fontsize=8
gr()
pyplot()
p = plot(time_dircol,stats_i["c_max"][2:end],yscale=:log10,label="DIRCOL",color=:blue,width=width,
    markershape=:circle,markerstrokecolor=:blue,markersize=markersize)
plot!(time_ilqr,stats_inf["c_max"],yscale=:log10,label="ALTRO",width=width,color=:darkorange2,
    markershape=:circle,markerstrokecolor=:darkorange2,markersize=markersize)
plot!(time_newton,c_max,yscale=:log10,ylim=[1e-9,1e-1],xlim=[0,10],label="ALTRO*",legend=:topright,xlabel="Time (s)",ylabel="Max Constraint Violation",color=:green,width=width,
    markershape=:circle,markerstrokecolor=:green,markersize=markersize,dpi=400,size=(500,250))
plot_vertical_lines!(p,[t_ilqr],label="projected Newton")
plot_vertical_lines!(p,[time_ilqr[stats_inf["max_mu_iteration (infeasible)"]]],label="max penalty",color=:red,style=:dot,legend=:bottom,
    tickfontsize=fontsize,legendfontsize=fontsize,guidefontsize=fontsize)

savefig(p,joinpath(IMAGE_DIR,"escape_newton.png"))

cost(solver,res_)
stats_i["cost"][end-3]
stats_inf["cost"][end]

# solver_truth = Solver(model, obj, N=601)
#
# Ns = [101,145,201,251,281]
# group = "escape"
# run_step_size_comparison(model, obj, U0, group, Ns, opts=opts, integrations=[:rk3,:ipopt],benchmark=false, infeasible=true, X0=X0, dt_truth=solver_truth.dt)
# plot_stat("runtime",group,legend=:topleft,["rk3","ipopt"],title="Escape",color=[:blue :darkorange2],size=(500,250),ylim=(0,50))
# savefig(joinpath(IMAGE_DIR,"escape_runtime.eps"))
# plot_stat("iterations",group,legend=:topleft,["rk3","ipopt"],title="Escape",size=(500,250))
# savefig(joinpath(IMAGE_DIR,"escape_iters.eps"))
# plot_stat("error",group,yscale=:log10,legend=:right,["rk3","ipopt"],title="Escape")
#
# Ns, data = load_data("runtime",["rk3","ipopt"],"escape")
# Ns, err = load_data("std",["rk3","ipopt"],"escape")
# plot(Ns,data[1],yerr=err[1],label="ALTRO",markershape=:circle,markersize=6,color=:darkorange2,markerstrokecolor=:darkorange2)
# plot!(Ns,data[2],yerr=err[2],label="Ipopt",markershape=:circle,markersize=6,color=:blue,markerstrokecolor=:blue)
