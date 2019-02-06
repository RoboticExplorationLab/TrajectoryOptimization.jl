using Plots
include("N_plots.jl")

# Model and Objective
model, obj = Dynamics.dubinscar_parallelpark
obj_uncon = UnconstrainedObjective(obj)
obj = update_objective(obj,tf=2.)

# Solver Options
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

# Solve original problem
N = 51
solver = Solver(model, obj, N=N, opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(solver.model.m,solver_con_box.N-1)
res, stats = solve(solver,U0)

# Mintime
opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-4
opts.cost_tolerance_intermediate = 1e-4
opts.max_dt = 0.2
opts.min_dt = 1e-3
opts.minimum_time_dt_estimate = tf/(N-1)
opts.constraint_tolerance = 0.001 # 0.005
opts.R_minimum_time = 2. #15.0 #13.5 # 12.0
opts.constraint_decrease_ratio = .25
opts.penalty_scaling = 10.0
opts.outer_loop_update_type = :individual
opts.iterations = 1000
opts.iterations_outerloop = 30 # 20
opts.square_root = true

ipopt_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)

obj_mintime = update_objective(obj_con_box,tf=:min, u_min=u_min, u_max=u_max)

solver_mintime = Solver(model, obj_mintime, integration=integration, N=N, opts=opts)
results_mintime, stats_mintime = solve(solver_mintime,to_array(results_con_box.U))

X0 = rollout(solver,U0)
res_d, stats_d = solve_dircol(solver_mintime,X0,U0,options=ipopt_options)
stats_d["info"]

T = TrajectoryOptimization.total_time(solver,res)
T_min = TrajectoryOptimization.total_time(solver_mintime,results_mintime)
T_d = TrajectoryOptimization.total_time(solver_mintime,res_d)

plot(to_array(res.U)[1:2,1:solver_mintime.N-1]',labels="original",linewidth=2,linestyle=:dash,xlabel="knot point",ylabel="control")
plot!(to_array(results_mintime.U)[1:2,1:solver_mintime.N-1]',linestyle=:solid,color=[1 2],labels="mintime",linewidth=2)
plot!(res_d.U[1:2,1:solver_mintime.N-1]',linestyle=:dot,color=[1 2],labels="dircol",linewidth=2)

plot(to_array(res.X)[1:3,:]',linewidth=2,linestyle=:dash,xlabel="knot point",ylabel="state",labels=["x" "y" "theta"])
plot!(to_array(results_mintime.X)[1:3,:]',linewidth=2,color=[1 2 3],legend=:topleft,labels="")
plot!(res_d.X[1:3,:]',linewidth=2,color=[1 2 3],legend=:topleft,labels="",style=:dot)

t = get_time(solver)
t_mintime = get_time(solver_mintime,results_mintime)
plot(t,to_array(res.X)[1:3,:]',linewidth=2,linestyle=:dash,xlabel="time (s)",ylabel="state",labels=["x" "y" "theta"])
plot!(t_mintime,to_array(results_mintime.X)[1:3,1:N-1]',linewidth=2,color=[1 2 3],legend=:topleft,labels="")

p = plot()
plot_trajectory!(res,label="original",width=2,xlabel="x",ylabel="y")
plot_trajectory!(results_mintime,labels="mintime",width=2,legend=:topleft)
plot_trajectory!(res_d,labels="dircol",width=2,legend=:topleft)
