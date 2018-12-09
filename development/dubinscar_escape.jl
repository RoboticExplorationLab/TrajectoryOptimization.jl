include("paper/N_plots.jl")
Random.seed!(7)

model, = TrajectoryOptimization.Dynamics.dubinscar!
n, m = model.n,model.m

# Objective
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

u_bnd = 6
obj_con_obstacles = TrajectoryOptimization.ConstrainedObjective(obj,cI=cI,u_min=-u_bnd,u_max=u_bnd)

opts = SolverOptions()
opts.resolve_feasible = false
opts.R_infeasible = 1.
opts.live_plotting = false
opts.verbose = false
N = 101

solver = Solver(model, obj_con_obstacles, integration=:rk3, N=N, opts=opts)
solver_foh = Solver(model, obj_con_obstacles, integration=:rk3_foh, N=N, opts=opts)
solver_dircol = Solver(model, obj_con_obstacles, integration=:rk3_foh, N=N)

# -Initial state and control trajectories
X_guess = [2.5 2.5 0.;4. 5. .785;5. 6.25 0.;7.5 6.25 -.261;9 5. -1.57;7.5 2.5 0.]
X0 = interp_rows(N, tf, Array(X_guess'))
U0 = rand(m,N)

@time res, stats = TrajectoryOptimization.solve(solver,X0,U0)
@time res_foh, stats_foh = TrajectoryOptimization.solve(solver_foh,X0,U0)


solver_dircol.opts.verbose = false
res_d,stats_d = solve_dircol(solver_dircol, X0, U0)


# Escape
plt = plot(title="Escape",aspect_ratio=:equal)
plot_obstacles(circles)
plot_trajectory!(to_array(res.X),width=2,color=:blue,label="rk3",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(res_foh.X),width=2,color=:purple,label="rk3_foh",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot!(X0[1,:],X0[2,:],label="Infeasible Initialization",width=1,color=:purple,linestyle=:dash)
plot_trajectory!(res_d.X,width=2,color=:black,label="DIRCOL",linestyle=:dash)

eval_f, = gen_usrfun_ipopt(solver_dircol,:hermite_simpson)
var_zoh = DircolVars(res)
var_foh = DircolVars(res_foh)
eval_f(var_zoh.Z)
eval_f(var_foh.Z)
eval_f(res_d.Z)

@show stats["iterations"]
@show stats_foh["iterations"]


group = "dubinscar/escape"
solver_truth = Solver(model, obj_con_obstacles, N=501, integration=:rk3_foh)
dt_truth = solver_truth.dt
run_dircol_truth(solver_truth, Array(res_d.X), Array(res_d.U), group::String)

run_step_size_comparison(model, obj_con_obstacles, U0, group::String, [101,201,301]; integrations=[:rk3,:rk3_foh],dt_truth=dt_truth,opts=opts,infeasible=true,X0=X0)
plot_stat("runtime",group,legend=:topleft,title="Constrained Parallel Park")
plot_stat("iterations",group,legend=:left,title="Constrained Parallel Park")
plot_stat("error",group,yscale=:log10,legend=:right,title="Constrained Parallel Park")
plot_stat("c_max",group,yscale=:log10,legend=:right,title="Constrained Parallel Park")
