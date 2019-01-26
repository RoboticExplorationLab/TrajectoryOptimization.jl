import TrajectoryOptimization: gen_usrfun_ipopt
using Random, LinearAlgebra, Plots
using BenchmarkTools, Statistics
Random.seed!(7)
include("N_plots.jl")

# Solver Options
dt = 0.01
integration = :rk3
method = :hermite_simpson

function convergence_plot(stat_i,stat_d;kwargs...)
    plot(log.(abs.(stat_i["cost"])),width=2,label="iLQR")
    plot!(log.(abs.(stat_d["cost"])), ylabel="Cost (abs log)",xlabel="iterations",width=2,label="DIRCOL"; kwargs...)
end

###################
## Parallel Park ##
###################

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5
opts.resolve_feasible = false
opts.outer_loop_update_type = :default
opts.use_nesterov = true
opts.penalty_scaling = 200
opts.penalty_initial = 1
opts.R_infeasible = 10

dircol_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)

# Set up model, objective, and solver
model, = TrajectoryOptimization.Dynamics.dubinscar
n, m = model.n,model.m

x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con_box = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

solver_uncon  = Solver(model, obj, integration=integration, dt=dt, opts=opts)
solver_con_box = Solver(model, obj_con_box, integration=integration, dt=dt, opts=opts)
solver_nn = Solver(model, obj_con_box, integration=integration, dt=dt, opts=opts)
solver_nn.opts.use_nesterov = false

U0 = rand(solver_uncon.model.m,solver_uncon.N)
X0 = line_trajectory(solver_con_box)
X0_rollout = rollout(solver_uncon, U0)

@time results_uncon, stats_uncon = TrajectoryOptimization.solve(solver_uncon,U0)

solver_con = Solver(model, obj_con_box, integration=integration, dt=dt, opts=opts)
solver_con.opts.use_nesterov = false
solver_con.opts.square_root = false
solver_con.opts.penalty_scaling = 100
solver_con.opts.penalty_initial = 0.01
solver_con.opts.outer_loop_update_type = :feedback
solver_con.opts.cost_tolerance_intermediate = 1e-3
solver_con.opts.constraint_tolerance_intermediate = 1
solver_con.opts.use_penalty_burnin = false
solver_con.opts.verbose = false
@time results_con_box, stats_con_box = solve(solver_con,U0)
stats_con_box["c_max"][end]

solver_con_n = Solver(solver_con)
solver_con_n.opts.use_nesterov = true
solver_con_n.opts.square_root = true
results_n, stats_n = solve(solver_con_n,U0)

plot(stats_con_box["c_max"],yscale=:log10)
plot!(stats_n["c_max"])

solver_inf = Solver(model, obj_con_box, integration=integration, dt=dt, opts=opts)
solver_inf.opts.use_nesterov = false
solver_inf.opts.penalty_scaling = 100
solver_inf.opts.penalty_initial = 0.001
solver_inf.opts.outer_loop_update_type = :feedback
solver_inf.opts.cost_tolerance_intermediate = 1e-3
solver_inf.opts.constraint_tolerance_intermediate = 1
solver_inf.opts.use_penalty_burnin = false
solver_inf.opts.verbose = false
solver_inf.opts.R_infeasible = 10
@time res_inf, stats_inf = TrajectoryOptimization.solve(solver_inf,X0,U0)
stats_inf["iterations"]

@time res_d, stats_d = TrajectoryOptimization.solve_dircol(solver_con_box, X0_rollout, U0, options=dircol_options)


# Parallel Park (boxed)
plt = plot(title="Parallel Park")#,aspect_ratio=:equal)
plot!(x_min[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(x_max[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_min[2]*ones(1000),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_max[2]*ones(1000),color=:red,width=2,label="")
plot_trajectory!(to_array(results_uncon.X),width=2,color=:blue,label="Unconstrained")
plot_trajectory!(to_array(results_con_box.X),width=2,color=:green,label="Constrained",legend=:bottomright)
plot_trajectory!(to_array(res_inf.X),width=2,color=:yellow,label="Constrained (infeasible)",legend=:bottomright)
plot_trajectory!(res_d.X,width=2,color=:black,linestyle=:dash,label="DIRCOL")

# Stats comparison
eval_f, = gen_usrfun_ipopt(solver_con_box,method)
res_i = DircolVars(results_con_box)
stats_con_box["runtime"]
stats_d["runtime"]
stats_con_box["iterations"]
stats_d["iterations"]
eval_f(res_i.Z)
eval_f(res_d.Z)
TrajectoryOptimization._cost(solver_con_box,results_con_box)
plot(stats_uncon["gradient_norm"][10:end])
stats_uncon["gradient_norm"]

# Convergence Behavior
convergence_plot(stats_con_box,stats_d)
plot!(log.(abs.(stats_inf["cost"])),linewidth=2,label="iLQR (infeasible)")
convergence_rate(stats_uncon,tail=0.5,plot_fit=true)
convergence_rate(stats_con_box,tail=0.5,plot_fit=true)
convergence_rate(stats_inf,tail=0.5,plot_fit=true)

p = plot(stats_con_box["gradient_norm"],yscale=:log10,label="no nesterov",color=:blue,xlabel="iterations",ylabel="gradient",title="Constrained Parallel Park")
ylim = collect(ylims(p))
plot_vertical_lines!(stats_con_box["outer_updates"],ylim,linecolor=:blue,linestyle=:dash,label="")
plot!(stats_n["gradient_norm"],yscale=:log10,label="nesterov",linecolor=:red,legend=:bottomleft)
plot_vertical_lines!(stats_n["outer_updates"],ylim,linecolor=:red,linestyle=:dash,label="")


# STEP SIZE COMPARISONS #
# Unconstrained
Ns = [21,31,51,81,101,201,301]
disable_logging(Logging.Debug)
group = "dubinscar/parallelpark/unconstrained"
run_step_size_comparison(model, obj, U0, group, Ns, integrations=[:midpoint,:rk3],dt_truth=1e-3,opts=opts,benchmark=true)
plot_stat("runtime",group,legend=:bottomright,title="Unconstrained Parallel Park")
plot_stat("iterations",group,legend=:bottom,title="Unconstrained Parallel Park")
plot_stat("error",group,yscale=:log10,legend=:right,title="Unconstrained Parallel Park")

# Constrained
Ns = [21,31,51,81,151,201,301]
disable_logging(Logging.Debug)
group = "dubinscar/parallelpark/constrained"
run_step_size_comparison(model, obj_con_box, U0, group, Ns, integrations=[:midpoint,:rk3],dt_truth=1e-3,opts=solver_con.opts)
plot_stat("runtime",group,legend=:topleft,title="Constrained Parallel Park")
plot_stat("iterations",group,legend=:topright,title="Constrained Parallel Park")
plot_stat("error",group,yscale=:log10,legend=:topright,title="Constrained Parallel Park")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Constrained Parallel Park")

# Infeasible
Ns = [21,31,51,81,151,201,301]
disable_logging(Logging.Debug)
group = "dubinscar/parallelpark/infeasible"
run_step_size_comparison(model, obj_con_box, U0, group, Ns, integrations=[:midpoint,:rk3],dt_truth=1e-3,opts=solver_inf.opts, infeasible=true)
plot_stat("runtime",group,legend=:topleft,title="Constrained Parallel Park (infeasible)")
plot_stat("iterations",group,legend=:topright,title="Constrained Parallel Park (infeasible)")
plot_stat("error",group,yscale=:log10,legend=:topright,title="Constrained Parallel Park (infeasible)")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Constrained Parallel Park (infeasible)")

dt_truth = 1e-3
solver_truth = Solver(model, obj, dt=dt_truth)
X_truth, U_truth = get_dircol_truth(solver_truth,X0,U0,group)[2:3]
interp(t) = TrajectoryOptimization.interpolate_trajectory(solver_truth, X_truth, U_truth, t)

opts.use_nesterov = false
err, err_final, stats = run_Ns(model, obj_con_box, X0, U0, Ns, interp, opts=opts)
opts.use_nesterov = true
err_n, err_final_n, stats_n = run_Ns(model, obj_con_box, X0, U0, Ns, interp, opts=opts)
iters = [[stat["iterations"] for stat in sts] for sts in [stats,stats_n]]
runtime = [[stat["runtime"] for stat in sts] for sts in [stats,stats_n]]
plot_vals(Ns,iters,["default","nesterov"],"iterations")
plot_vals(Ns,runtime,["default","nesterov"],"runtime")



########################
## Obstacle Avoidance ##
########################

opts.penalty_initial = 0.1

x0 = [0.0;0.0;0.]
xf = [10.0;10.0;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

u_min = [-100.; -100.]
u_max = [100.; 100.]
x_min = [-Inf; -Inf; -Inf]
x_max = [Inf; Inf; Inf]

n_circles = 3
r = 1.0
circles = ((2.5,2.5,r),(5.,5.,r),(7.5,7.5,r))

function cI(c,x,u)
    for i = 1:n_circles
        c[i] = circle_constraint(x,circles[i][1],circles[i][2],circles[i][3])
    end
    c
end

obj_con_obstacles = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max,cI=cI)
obj_con_obstacles_control = TrajectoryOptimization.ConstrainedObjective(obj,u_min=u_min, u_max=u_max,x_min=x_min,x_max=x_max,cI=cI)

solver_uncon_obstacles = Solver(model, obj, integration=integration, dt=dt, opts=opts)
solver_con_obstacles = Solver(model, obj_con_obstacles, integration=integration, dt=dt, opts=opts)
solver_con_obstacles.opts.penalty_scaling = 10
solver_con_obstacles.opts.use_nesterov = false

# -Initial state and control trajectories
U0 = rand(solver_uncon.model.m,solver_uncon_obstacles.N)
X0 = line_trajectory(solver_con_obstacles)
X0_rollout = rollout(solver_con_obstacles,U0)

@time results_uncon_obstacles, stats_uncon_obstacles = TrajectoryOptimization.solve(solver_uncon_obstacles,U0)
@time results_con_obstacles, stats_con_obstacles = TrajectoryOptimization.solve(solver_con_obstacles,U0)
results_inf, stats_inf = TrajectoryOptimization.solve(solver_con_obstacles,X0,U0)
res_d, stats_d = solve_dircol(solver_con_obstacles,X0_rollout,U0,options=dircol_options)


# Obstacle Avoidance
plt = plot(title="Obstacle Avoidance")
plot_obstacles(circles)
plot_trajectory!(to_array(results_uncon_obstacles.X),width=2,color=:blue,label="Unconstrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(results_con_obstacles.X),width=2,color=:green,label="Constrained")
plot_trajectory!(to_array(results_con_obstacles.X),width=2,color=:orange,label="Infeasible")
plot_trajectory!(res_d.X,width=2,color=:black,linestyle=:dash,label="DIRCOL",legend=:bottomright)
display(plt)

group
solver_truth = Solver(model,obj_con_obstacles,dt=1e-3)
_,X_truth,U_truth = get_dircol_truth(solver_truth,X0,U0,group)
plot_trajectory!(X_truth,width=2,color=:grey,linestyle=:dash,label="DIRCOL (truth)",legend=:bottomright)
res_truth = DircolVars(X_truth,U_truth)


# Stats comparison
eval_f_truth, = gen_usrfun_ipopt(solver_truth,method)
eval_f, = gen_usrfun_ipopt(solver_con_obstacles,method)
res_i = DircolVars(to_array(results_con_obstacles.X),to_array(results_con_obstacles.U))
res_inf = DircolVars(results_inf)
stats_con_obstacles["runtime"]
stats_d["runtime"]
stats_con_obstacles["iterations"]
stats_d["iterations"]
convergence_plot(stats_con_obstacles,stats_d)
plot!(stats_inf["cost"],width=2,label="Infeasible")
eval_f(res_i.Z)
eval_f(res_d.Z)
eval_f(res_inf.Z)
eval_f_truth(res_truth.Z)
TrajectoryOptimization._cost(solver_con_obstacles,results_con_obstacles)
TrajectoryOptimization._cost(solver_con_obstacles,results_inf)

# KNOT POINT COMPARISON
Ns = [21,41,51,81,101,201,301]
opts.penalty_scaling = 10
opts.use_nesterov = false
disable_logging(Logging.Debug)
group = "dubinscar/obstacles/constrained"
run_step_size_comparison(model, obj_con_obstacles, U0, group, Ns, integrations=[:midpoint,:rk3],dt_truth=1e-3,opts=opts)
plot_stat("runtime",group,legend=:topleft,title="Obstacle Avoidance")
plot_stat("iterations",group,legend=:left,title="Obstacle Avoidance")
plot_stat("error",group,yscale=:log10,legend=:topright,title=" Obstacle Avoidance")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Obstacle Avoidance")

Ns = [21,41,51,81,101,201,301]
opts.penalty_scaling = 10
opts.use_nesterov = false
disable_logging(Logging.Debug)
group = "dubinscar/obstacles/infeasible"
run_step_size_comparison(model, obj_con_obstacles, U0, group, Ns, integrations=[:midpoint,:rk3],dt_truth=1e-3,opts=opts,infeasible=true)
plot_stat("runtime",group,legend=:topleft,title="Obstacle Avoidance")
plot_stat("iterations",group,legend=:left,title="Obstacle Avoidance")
plot_stat("error",group,yscale=:log10,legend=:topright,title=" Obstacle Avoidance")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Obstacle Avoidance")

# Nesterov comparison
opts.use_nesterov = false
err, err_final, stats = run_Ns(model, obj_con_obstacles, X0, U0, Ns, interp, opts=opts, infeasible=true)
opts.use_nesterov = true
err_n, err_final_n, stats_n = run_Ns(model, obj_con_obstacles, X0, U0, Ns, interp, opts=opts, infeasible=true)
iters = [[stat["iterations"] for stat in sts] for sts in [stats,stats_n]]
runtime = [[stat["runtime"] for stat in sts] for sts in [stats,stats_n]]
plot_vals(Ns,iters,["default","nesterov"],"iterations")
plot_vals(Ns,runtime,["default","nesterov"],"runtime")

############
## Escape ##
############

x0 = [2.5;2.5;0.]
xf = [7.5;2.5;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

r = 0.5
s1 = 30; s2 = 50; s3 = 15
circles = []
for i in range(0,stop=5,length=s1)
    push!(circles,(0.,i,r))
end
for i in range(0,stop=5,length=s1)
    push!(circles,(5.,i,r))
end
for i in range(0,stop=5,length=s1)
    push!(circles,(10.,i,r))
end
for i in range(0,stop=10,length=s2)
    push!(circles,(i,0.,r))
end
for i in range(0,stop=3,length=s3)
    push!(circles,(i,5.,r))
end
for i in range(5,stop=8,length=s3)
    push!(circles,(i,5.,r))
end

n_circles = 3*s1 + s2 + 2*s3

# plt = plot(title="Escape",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
# plot_obstacles(circles,:red)
# display(plt)

function cI(c,x,u)
    for i = 1:n_circles
        c[i] = circle_constraint(x,circles[i][1],circles[i][2],circles[i][3])
    end
    c
end

obj_con_obstacles = TrajectoryOptimization.ConstrainedObjective(obj,cI=cI,u_min=-10,u_max=10)

solver_uncon_obstacles = Solver(model, obj, dt=dt, opts=opts)
solver_con_obstacles = Solver(model, obj_con_obstacles, dt=dt, opts=opts)
solver_con_obstacles.opts.R_infeasible = 1.0
n,m,N = get_sizes(solver_con_obstacles)

# -Initial state and control trajectories
X_guess = [2.5 2.5 0.;4. 5. .785;5. 6.25 0.;7.5 6.25 -.261;9 5. -1.57;7.5 2.5 0.]
X0 = TrajectoryOptimization.interp_rows(N,tf,Array(X_guess'))
# X0 = line_trajectory(solver_uncon)
U0 = ones(m,N)

# @time results_uncon_obstacles, stats_uncon_obstacles = TrajectoryOptimization.solve(solver_uncon_obstacles,U0)
# @time results_con_obstacles, stats_con_obstacles = TrajectoryOptimization.solve(solver_con_obstacles,U0)
solver = Solver(model, obj_con_obstacles, N=101, opts=SolverOptions())
solver.opts.R_infeasible = 1
solver.opts.square_root
solver.opts.resolve_feasible = false
solver.opts.cost_tolerance = 1e-6
solver.opts.cost_tolerance_intermediate = 1e-3
solver.opts.constraint_tolerance = 1e-5
solver.opts.constraint_tolerance_intermediate = 0.01
solver.opts.penalty_scaling = 50
solver.opts.penalty_initial = 10
solver.opts.outer_loop_update_type = :default
solver.opts.use_penalty_burnin = false
solver.opts.use_nesterov = false
solver.opts.live_plotting = true
solver.opts.verbose = true
@time res_inf, stats_inf = solve(solver,X0,U0)
stats_inf["iterations"]

res_d,stats_d = solve_dircol(solver,X0,U0,options=dircol_options)

# Escape
plt = plot(title="Escape",aspect_ratio=:equal)
plot_obstacles(circles)
# plot_trajectory!(to_array(results_uncon_obstacles.X),width=2,color=:blue,label="Unconstrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
# plot_trajectory!(to_array(results_con_obstacles.X),width=2,color=:green,label="Constrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(res_inf.X),width=2,color=:purple,label="Infeasible",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(res_d.X,width=2,color=:yellow,label="DIRCOL",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot!(X0[1,:],X0[2,:],label="Infeasible Initialization",width=1,color=:purple,linestyle=:dash)
display(plt)


# Dircol Truth
group = "dubinscar/escape/infeasible"
solver_truth = Solver(model,obj_con_obstacles,N=1001)
solve_dircol(solver_truth,Array(res_d.X), Array(res_d.U))
run_dircol_truth(solver_truth, Array(res_d.X), Array(res_d.U), group::String)

Ns = [101,151,201,251,301]
disable_logging(Logging.Debug)
run_step_size_comparison(model, obj_con_obstacles, U0, group, Ns, integrations=[:midpoint,:rk3],dt_truth=solver_truth.dt,opts=solver.opts,infeasible=true,X0=X0)
plot_stat("runtime",group,legend=:topleft,title="Obstacle Avoidance",yscale=:log10)
plot_stat("iterations",group,legend=:right,title="Obstacle Avoidance",yscale=:log10)
plot_stat("error",group,yscale=:log10,legend=:topright,title=" Obstacle Avoidance")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Obstacle Avoidance")



X_truth, U_truth = get_dircol_truth(solver_truth,X0,U0,group)[2:3]
interp(t) = TrajectoryOptimization.interpolate_trajectory(solver_truth, X_truth, U_truth, t)

opts_no_nesterov = copy(solver.opts)
opts_no_nesterov.use_nesterov = false
opts_no_nesterov.outer_loop_update_type = :feedback
err, err_final, stats = run_Ns(model, obj_con_box, X0, U0, Ns, interp, opts=opts_no_nesterov)
opts_no_nesterov.use_nesterov = true
err_n, err_final_n, stats_n = run_Ns(model, obj_con_box, X0, U0, Ns, interp, opts=opts_no_nesterov)
iters = [[stat["iterations"] for stat in sts] for sts in [stats,stats_n]]
runtime = [[stat["runtime"] for stat in sts] for sts in [stats,stats_n]]
plot_vals(Ns,iters,["default","nesterov"],"iterations")
plot_vals(Ns,runtime,["default","nesterov"],"runtime")


plot(stats[2]["gradient_norm"],yscale=:log10)
plot!(stats_n[2]["gradient_norm"])
