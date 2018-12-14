Random.seed!(7)
include("N_plots.jl")

# Solver Options
dt = 0.01
integration = :rk3_foh
method = :hermite_simpson
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false

function convergence_plot(stat_i,stat_d)
    plot(stat_i["cost"],width=2,label="iLQR")
    plot!(stat_d["cost"], ylim=[0,10], ylabel="Cost",xlabel="iterations",width=2,label="DIRCOL")
end

###################
## Parallel Park ##
###################

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_intermediate_tolerance = 1e-5

# Set up model, objective, and solver
model, = TrajectoryOptimization.Dynamics.dubinscar
n, m = model.n,model.m

x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con_box = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

solver_uncon  = Solver(model, obj, integration=integration, dt=dt, opts=opts)
solver_con_box = Solver(model, obj_con_box, integration=integration, N=301, opts=opts)

U0 = rand(solver_uncon.model.m,solver_uncon.N)
X0 = line_trajectory(solver_con_box)
X0_rollout = rollout(solver_uncon, U0)

@time results_uncon, stats_uncon = TrajectoryOptimization.solve(solver_uncon,U0)
@time results_con_box, stats_con_box = TrajectoryOptimization.solve(solver_con_box,U0)
res_inf, stats_inf = TrajectoryOptimization.solve(solver_con_box,X0,U0)
res_d, stats_d = TrajectoryOptimization.solve_dircol(solver_con_box, X0_rollout, U0)


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
# plot_trajectory!(X_truth,width=2,color=:yellow,linestyle=:dash,label="DIRCOL (truth)")
display(plt)

# Stats comparison
eval_f, = gen_usrfun_ipopt(solver_con_box,method)
res_i = DircolVars(to_array(results_con_box.X),to_array(results_con_box.U))
stats_con_box["runtime"]
stats_d["runtime"]
stats_con_box["iterations"]
stats_d["iterations"]
convergence_plot(stats_con_box,stats_d)
eval_f(res_i.Z)
eval_f(res_d.Z)
_cost(solver_con_box,results_con_box)


# STEP SIZE COMPARISONS #
# Unconstrained
Ns = [21,41,51,81,101,201,401,501,801,1001]
disable_logging(Logging.Debug)
group = "dubinscar/parallelpark/unconstrained"
run_step_size_comparison(model, obj, U0, group, Ns, integrations=[:rk3,:rk3_foh],dt_truth=1e-3,opts=opts)
plot_stat("runtime",group,legend=:topleft,title="Unconstrained Parallel Park")
plot_stat("iterations",group,legend=:topright,title="Unconstrained Parallel Park")
plot_stat("error",group,[:rk3,:rk3_foh],yscale=:log10,legend=:right,title="Unconstrained Parallel Park")

# Constrained
Ns = [21,41,51,81,101,201,401,501,801,1001]
disable_logging(Logging.Debug)
group = "dubinscar/parallelpark/constrained"
run_step_size_comparison(model, obj_con_box, U0, group, Ns, integrations=[:rk3,:rk3_foh],dt_truth=1e-3,opts=opts)
plot_stat("runtime",group,legend=:topleft,title="Constrained Parallel Park")
plot_stat("iterations",group,legend=:left,title="Constrained Parallel Park")
plot_stat("error",group,yscale=:log10,legend=:right,title="Constrained Parallel Park")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Constrained Parallel Park")

# Infeasible
opts = SolverOptions()
opts.cost_tolerance = 1e-6
opts.cost_intermediate_tolerance = 1e-5
opts.resolve_feasible = false
Ns = [21,41,51,81,101,201,401,501,801,1001]
disable_logging(Logging.Debug)
group = "dubinscar/parallelpark/infeasible"
run_step_size_comparison(model, obj_con_box, U0, group, Ns, integrations=[:rk3,:rk3_foh],dt_truth=1e-3,opts=opts, infeasible=true)
plot_stat("runtime",group,legend=:topleft,title="Constrained Parallel Park (infeasible)")
plot_stat("iterations",group,legend=:left,title="Constrained Parallel Park (infeasible)")
plot_stat("error",group,yscale=:log10,legend=:right,title="Constrained Parallel Park (infeasible)")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Constrained Parallel Park (infeasible)")


####################
#   MINIMUM TIME   $
####################
opts = SolverOptions()
opts.verbose = true
opts.cost_tolerance = 1e-6
opts.cost_intermediate_tolerance = 1e-5
opts.minimum_time_tf_estimate = 2.0
opts.gradient_tolerance = 1e-10
opts.live_plotting = true

u_bnd = 2
N = 51
obj_mintime = update_objective(obj_con_box,tf=:min, u_min=-u_bnd, u_max=u_bnd)
solver_min = Solver(model, obj_mintime, N=N, integration=:rk3, opts=opts)

U0 = zeros(m,N)
res,stats = solve(solver_min, U0)


########################
## Obstacle Avoidance ##
########################

x0 = [0.0;0.0;0.]
xf = [10.0;10.0;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

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
solver_con_obstacles_control = Solver(model, obj_con_obstacles_control, integration=integration, dt=dt, opts=opts)

# -Initial state and control trajectories
U0 = rand(solver_uncon.model.m,solver_uncon.N)
X0 = line_trajectory(solver_con_obstacles)
X0_rollout = rollout(solver_con_obstacles,U0)

@time results_uncon_obstacles, stats_uncon_obstacles = TrajectoryOptimization.solve(solver_uncon_obstacles,U0)
println("Final state (unconstrained)-> pos: $(results_uncon_obstacles.X[end][1:3]), goal: $(solver_uncon_obstacles.obj.xf[1:3])\n Cost: $(stats_uncon_obstacles["cost"][end])\n Iterations: $(stats_uncon_obstacles["iterations"])\n Outer loop iterations: $(stats_uncon_obstacles["major iterations"])\n ")

@time results_con_obstacles, stats_con_obstacles = TrajectoryOptimization.solve(solver_con_obstacles,U0)
println("Final state (constrained)-> pos: $(results_con_obstacles.X[end][1:3]), goal: $(solver_con_obstacles.obj.xf[1:3])\n Cost: $(stats_con_obstacles["cost"][end])\n Iterations: $(stats_con_obstacles["iterations"])\n Outer loop iterations: $(stats_con_obstacles["major iterations"])\n Max violation: $(stats_con_obstacles["c_max"][end])\n Max μ: $(maximum([to_array(results_con_obstacles.μ)[:]; results_con_obstacles.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_con_obstacles.λ)[:]; results_con_obstacles.λN[:]])))\n")

results_inf, stats_inf = TrajectoryOptimization.solve(solver_con_obstacles,X0,U0)
res_d, stats_d = solve_dircol(solver_con_obstacles,X0,U0)


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
_cost(solver_con_obstacles,results_con_obstacles)
_cost(solver_con_obstacles,results_inf)

# KNOT POINT COMPARISON
Ns = [21,41,51,81,101,201,401,501,801,1001]
disable_logging(Logging.Debug)
group = "dubinscar/obstacles/constrained"
run_step_size_comparison(model, obj_con_obstacles, U0, group, Ns, integrations=[:rk3,:rk3_foh],dt_truth=1e-3,opts=opts)
plot_stat("runtime",group,legend=:topleft,title="Obstacle Avoidance")
plot_stat("iterations",group,legend=:left,title="Obstacle Avoidance")
plot_stat("error",group,yscale=:log10,legend=:topright,title=" Obstacle Avoidance")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Obstacle Avoidance")


############
## Escape ##
############

x0 = [2.5;2.5;0.]
xf = [7.5;2.5;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

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

obj_con_obstacles = TrajectoryOptimization.ConstrainedObjective(obj,cI=cI)

solver_uncon_obstacles = Solver(model, obj, integration=integration, dt=dt, opts=opts)
solver_con_obstacles = Solver(model, obj_con_obstacles, integration=integration, dt=dt, opts=opts)
solver_con_obstacles.opts.R_infeasible = 1.0
# -Initial state and control trajectories
X_guess = [2.5 2.5 0.;4. 5. .785;5. 6.25 0.;7.5 6.25 -.261;9 5. -1.57;7.5 2.5 0.]
X0 = interp_rows(solver_uncon.N,tf,Array(X_guess'))
# X0 = line_trajectory(solver_uncon)
U0 = rand(solver_uncon.model.m,solver_uncon.N)

@time results_uncon_obstacles, stats_uncon_obstacles = TrajectoryOptimization.solve(solver_uncon_obstacles,U0)
println("Final state (unconstrained)-> pos: $(results_uncon_obstacles.X[end][1:3]), goal: $(solver_uncon_obstacles.obj.xf[1:3])\n Cost: $(stats_uncon_obstacles["cost"][end])\n Iterations: $(stats_uncon_obstacles["iterations"])\n Outer loop iterations: $(stats_uncon_obstacles["major iterations"])\n ")

@time results_con_obstacles, stats_con_obstacles = TrajectoryOptimization.solve(solver_con_obstacles,U0)
println("Final state (constrained)-> pos: $(results_con_obstacles.X[end][1:3]), goal: $(solver_con_obstacles.obj.xf[1:3])\n Cost: $(stats_con_obstacles["cost"][end])\n Iterations: $(stats_con_obstacles["iterations"])\n Outer loop iterations: $(stats_con_obstacles["major iterations"])\n Max violation: $(stats_con_obstacles["c_max"][end])\n Max μ: $(maximum([to_array(results_con_obstacles.μ)[:]; results_con_obstacles.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_con_obstacles.λ)[:]; results_con_obstacles.λN[:]])))\n")

@time results_con_obstacles_inf, stats_con_obstacles_inf = TrajectoryOptimization.solve(solver_con_obstacles,X0,U0)
println("Final state (infeasible + constrained)-> pos: $(results_con_obstacles_inf.X[end][1:3]), goal: $(solver_con_obstacles.obj.xf[1:3])\n Cost: $(stats_con_obstacles_inf["cost"][end])\n Iterations: $(stats_con_obstacles_inf["iterations"])\n Outer loop iterations: $(stats_con_obstacles_inf["major iterations"])\n Max violation: $(stats_con_obstacles_inf["c_max"][end])\n Max μ: $(maximum([to_array(results_con_obstacles_inf.μ)[:]; results_con_obstacles_inf.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_con_obstacles_inf.λ)[:]; results_con_obstacles_inf.λN[:]])))\n")

# Escape
plt = plot(title="Escape",aspect_ratio=:equal)
plot_obstacles(circles)
plot_trajectory!(to_array(results_uncon_obstacles.X),width=2,color=:blue,label="Unconstrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(results_con_obstacles.X),width=2,color=:green,label="Constrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(results_con_obstacles_inf.X),width=2,color=:purple,label="Infeasible",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot!(X0[1,:],X0[2,:],label="Infeasible Initialization",width=1,color=:purple,linestyle=:dash)
display(plt)
