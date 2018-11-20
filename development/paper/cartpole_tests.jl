# Set up the problem
model, obj0 = Dynamics.cartpole_analytical
n,m = model.n, model.m

obj = copy(obj0)
obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.0
u_bnd = 20
x_bnd = [0.6,Inf,Inf,Inf]
obj_c = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)
obj_min = update_objective(obj_c,tf=:min,c=1.,Q = obj.Q*0., Qf = obj.Qf*1)
dt = 0.1

# Params
N = 51
method = :hermite_simpson

# Initial Trajectory
U0 = ones(1,N)
X0 = line_trajectory(obj.x0,obj.xf,N)

X0_rollout = copy(X0)
solver = Solver(model,obj_c,N=N)
rollout!(X0_rollout,U0,solver)

# Dircol functions
eval_f, eval_g, eval_grad_f, eval_jac_g = gen_usrfun_ipopt(solver,method)

function comparison_plot(results,sol_ipopt;kwargs...)
    n = size(results.X,1)
    if n == 2 # pendulum
        state_labels = ["pos" "vel"]
    else
        state_labels = ["pos" "angle"]
    end
    time_dircol = range(0,stop=obj.tf,length=size(sol_ipopt.X,2))
    time_ilqr = range(0,stop=obj.tf,length=size(results.X,1))

    colors = [:blue :red]

    p_u = plot(time_ilqr,to_array(results.U)',width=2,label="iLQR",color=colors)
    plot!(time_dircol,sol_ipopt.U',width=2, label="IPOPT",color=colors,style=:dashdot,ylabel="control")

    p_x = plot(time_ilqr,to_array(results.X)[1:2,:]',width=2,label=state_labels,color=colors)
    plot!(time_dircol,sol_ipopt.X[1:2,:]',width=2, label="",color=colors,style=:dashdot,ylabel="state")

    p = plot(p_x,p_u,layout=(2,1),xlabel="time (s)"; kwargs...)

    return p
end

function convergence_plot(stat_i,stat_d)
    plot(stat_i["cost"])
    plot!(stat_d["cost"], ylim=[0,10], ylabel="Cost")
end

#####################################
#          UNCONSTRAINED            #
#####################################

# Solver Options
opts = SolverOptions()
opts.use_static = false
opts.cost_tolerance = 1e-6
opts.cost_intermediate_tolerance = 1e-4
opts.outer_loop_update = :default
opts.τ = 0.75

# iLQR
solver = Solver(model,obj,N=N,opts=opts,integration=:rk3_foh)
res_i, stat_i = solve(solver,U0)
plot(to_array(res_i.X)')
plot(to_array(res_i.U)')
max_violation(res_i)
cost(solver, res_i)
stat_i["runtime"]
stat_i["iterations"]
var_i = DircolVars(to_array(res_i.X),to_array(res_i.U))

# DIRCOL
res_d, stat_d = solve_dircol(solver, X0_rollout, U0; method=method)
plot(res_d.X')
plot(res_d.U')
stat_d["c_max"][end]
stat_d["cost"][end]
stat_d["runtime"]
stat_d["iterations"]

comparison_plot(res_i,res_d)
convergence_plot(stat_i,stat_d)

eval_f(var_i.Z)
eval_f(res_d.Z)

stat_i["cost"][50]
stat_d["cost"][50]

time_per_iter = stat_i["runtime"]/stat_i["iterations"]
time_per_iter = stat_d["runtime"]/stat_d["iterations"]


#####################################
#           CONSTRAINED             #
#####################################

# Solver Options
opts = SolverOptions()
opts.cost_tolerance = 1e-6
opts.cost_intermediate_tolerance = 1e-1
opts.outer_loop_update = :individual
opts.τ = .85

# iLQR
solver = Solver(model,obj_c,N=N,opts=opts,integration=:rk3_foh)
res_i, stat_i = solve(solver,U0)
plot(to_array(res_i.X)')
plot(to_array(res_i.U)')
max_violation(res_i)
cost(solver, res_i)
stat_i["runtime"]
stat_i["iterations"]

# DIRCOL
res_d, stat_d = solve_dircol(solver, X0_rollout, U0; method=:hermite_simpson)
plot(res_d.X')
plot(res_d.U')
stat_d["c_max"][end]
stat_d["cost"][end]
stat_d["runtime"]
stat_d["iterations"]

comparison_plot(res_i,res_d)
convergence_plot(stat_i,stat_d)

eval_f(var_i.Z)
eval_f(res_d.Z)

stat_i["cost"][40]
stat_d["cost"][40]

time_per_iter = stat_i["runtime"]/stat_i["iterations"]
time_per_iter = stat_d["runtime"]/stat_d["iterations"]


#####################################
#        INFEASIBLE START           #
#####################################

# Solver Options
opts = SolverOptions()
opts.cost_tolerance = 1e-8
opts.cost_intermediate_tolerance = 1e-1
# opts.constraint_tolerance = 1e-3
opts.outer_loop_update = :default
opts.τ = .85
opts.resolve_feasible = false

# iLQR
solver = Solver(model,obj_c,N=N,opts=opts,integration=:rk3)
res_i, stat_i = solve(solver,X0,U0)
plot(to_array(res_i.X)')
plot(to_array(res_i.U)')
max_violation(res_i)
_cost(solver, res_i)
stat_i["runtime"]
stat_i["iterations"]

# DIRCOL
res_d, stat_d = solve_dircol(solver, X0, U0; method=:hermite_simpson)
plot(res_d.X')
plot(res_d.U')
stat_d["c_max"][end]
stat_d["cost"][end]
stat_d["runtime"]
stat_d["iterations"]
stat_d["cost"]

comparison_plot(res_i,res_d)
convergence_plot(stat_i,stat_d)

eval_f(var_i.Z)
eval_f(res_d.Z)

stat_i["cost"][30]
stat_d["cost"][30]

time_per_iter = stat_i["runtime"]/stat_i["iterations"]
time_per_iter = stat_d["runtime"]/stat_d["iterations"]



#####################################
#          MINIMUM TIME             #
#####################################


# Solver Options
opts = SolverOptions()
opts.use_static = false
opts.max_dt = 0.25
opts.verbose = false
opts.cost_tolerance = 1e-4
opts.cost_intermediate_tolerance = 1e-3
opts.constraint_tolerance = 0.05
opts.outer_loop_update = :default
opts.R_minimum_time = 500
opts.resolve_feasible = false
opts.μ_initial_minimum_time_equality = 50.
opts.γ_minimum_time_equality = 15

# iLQR
solver = Solver(model,obj_min,N=N,opts=opts,integration=:rk3)
res_i, stat_i = solve(solver,U0)
plot(to_array(res_i.X)')
plot(to_array(res_i.U)')
max_violation(res_i)
_cost(solver, res_i)
stat_i["runtime"]
stat_i["iterations"]
total_time(solver, res_i)

# DIRCOL
solver.opts.verbose = false
res_d, stat_d = solve_dircol(solver, X0_rollout, U0; method=:hermite_simpson)
stat_d["info"]
plot(res_d.X')
plot(res_d.U')
stat_d["c_max"][end]
stat_d["cost"][end]
stat_d["runtime"]
stat_d["iterations"]
stat_d["cost"]
total_time(solver, res_d)

comparison_plot(res_i,res_d)
convergence_plot(stat_i,stat_d)

eval_f(var_i.Z)
eval_f(res_d.Z)

stat_i["cost"][30]
stat_d["cost"][30]

time_per_iter = stat_i["runtime"]/stat_i["iterations"]
time_per_iter = stat_d["runtime"]/stat_d["iterations"]

dt_i = [res_i.U[k][end]^2 for k = 1:N]
dt_d = res_d.U[end,:]

plot(dt_i)
plot!(dt_d)
