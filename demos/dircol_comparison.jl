using Plots
using TrajectoryOptimization
function comparison_plot(results,sol_ipopt;kwargs...)
    n = size(results.X,1)
    if n == 2 # pendulum
        state_labels = ["pos" "vel"]
    else
        state_labels = ["pos" "angle"]
    end
    time_dircol = linspace(0,obj.tf,size(sol_ipopt.X,2))
    time_ilqr = linspace(0,obj.tf,size(results.X,2))

    colors = [:blue :red]

    p_u = plot(time_ilqr,results.U',width=2,label="iLQR",color=colors,width=2)
    # plot!(time_dircol,sol_snopt.U',width=2, label="Snopt",color=colors,width=2,style=:dash)
    plot!(time_dircol,sol_ipopt.U',width=2, label="Ipopt",color=colors,width=2,style=:dashdot,ylabel="control")

    p_x = plot(time_ilqr,results.X[1:2,:]',width=2,label=state_labels,color=colors,width=2)
    # plot!(time_dircol,sol_snopt.X[1:2,:]',width=2, label="",color=colors,width=2,style=:dash)
    plot!(time_dircol,sol_ipopt.X[1:2,:]',width=2, label="",color=colors,width=2,style=:dashdot,ylabel="state")

    p = plot(p_x,p_u,layout=(2,1),xlabel="time (s)"; kwargs...)

    # p_x = plot(time_dircol,X_dircol[1:2,:]',width=2, label=state_labels.*" (DIRCOL)",color=colors)
    # plot!(time_ilqr,results.X[1:2,:]',width=2,label=state_labels.*" (iLQR)",color=colors, style=:dash,
    #     xlabel="time (s)", ylabel="States",legend=:topleft;kwargs...)
    return p
end


function ilqr_dircol_comparison(model, obj, method, name, X0=Array{Float64}(0,0);
    dt=0.01,mesh=[0.1,0.05,0.03,0.02],
    opts=SolverOptions(verbose=false,
                       eps_constraint=1e-6,
                       eps_intermediate=1e-4,
                       eps=1e-8))

    name *= "_" * string(method)

    # Initial state
    N,dt = TrajectoryOptimization.calc_N(obj.tf, dt)
    U = ones(1,N)*1
    X = zeros(model.n,N)
    if isempty(X0)
        X0 = line_trajectory(obj.x0, obj.xf, N)
    end

    # Feed iLQR to DIRCOL
    println("\niLQR → DIRCOL")
    solver = Solver(model,obj,dt=dt,integration=method,opts=opts)

    println("Solving iLQR...")
    time_i1 = @elapsed results, stats_i1 = solve(solver,X0,U)
    cost_i1 = cost(solver,results)

    println("Solving DIRCOL...")
    time_d1 = @elapsed sol, f_opt, stats_d1, objval = solve_dircol(solver, results.X, results.U, nlp=:ipopt, grads=:none, start=:warm)
    cost_d1 = cost(solver,sol)

    p1 = comparison_plot(results,sol,title=method)
    traj_diff = sol.X-results.X
    max_err1 = maximum(abs.(traj_diff))
    norm_err1 = vecnorm(traj_diff)

    # Feed DIRCOL to iLQR
    println("\nDIRCOL → iLQR")
    solver = Solver(model,obj,dt=dt,integration=method,opts=opts)

    println("Solving DIRCOL...")
    time_d2 = @elapsed sol2, f_opt, stats_d2, objval2 = solve_dircol(solver, X0, U, nlp=:ipopt, grads=:none)
    cost_d2 = cost(solver,sol2)

    println("Solving iLQR...")
    time_i2 = @elapsed results2, stats_i2 = solve(solver,Array(sol2.X),Array(sol2.U))
    cost_i2 = cost(solver,results2)

    p2 = comparison_plot(results2,sol2,title=method)
    traj_diff = sol2.X-results2.X
    max_err2 = maximum(abs.(traj_diff))
    norm_err2 = vecnorm(traj_diff)

    # File output
    println("Output to file...")
    folder = "logs"
    f = open(joinpath(folder,name * ".txt"),"w")
    col = [20,25,25]  # column widths
    create_row(s1,s2,s3) = join([rpad(s1,col[1]),rpad(s2,col[2]), rpad(s3,col[3])]," | ")
    header = create_row("\nStat","iLQR → DIRCOL","DIRCOL → iLQR")

    print_solver(solver,name,f)
    println(f,"\t final tolerance: $(solver.opts.eps)")
    println(f,"\t intermediate tolerance: $(solver.opts.eps_intermediate)")
    println(f,"\t constraint tolerance: $(solver.opts.eps_constraint)")
    println(f,header)
    println(f,repeat("-",sum(col)+3*3))
    println(f,create_row("iLQR runtime (sec)", time_i1, time_i2))
    println(f,create_row("DIRCOL runtime (sec)", time_d1, time_d2))
    println(f,create_row("iLQR cost", cost_i1, cost_i2))
    println(f,create_row("DIRCOL cost", cost_d1, cost_d2))
    println(f,create_row("iLQR iterations", stats_i1["iterations"], stats_i2["iterations"]))
    println(f,create_row("DIRCOL iterations", stats_d1["iterations"], stats_d2["iterations"]))
    println(f,create_row("Max error", max_err1, max_err2))
    println(f,create_row("Norm error", norm_err1, norm_err2))

    savefig(p1,joinpath(folder,name * "_iLQR_to_DIRCOL"))
    savefig(p2,joinpath(folder,name * "_DIRCOL_to_iLQR"))
    close(f)
end


#***************#
#   PENDULUM    #
#***************#

model = Model(Dynamics.pendulum_dynamics!,2,1)
obj0 = Dynamics.pendulum[2]
# model, obj = Dynamics.pendulum!
obj = copy(obj0)
obj.tf = 3.
obj.R .= [1e-2]
obj = ConstrainedObjective(obj)

# ilqr_dircol_comparison(model,obj,:midpoint,"Pendulum")
# ilqr_dircol_comparison(model,obj,:rk3_foh,"Pendulum")


#***************#
#   CART POLE   #
#***************#
model, obj0 = Dynamics.cartpole_analytical

obj = copy(obj0)
obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.

ilqr_dircol_comparison(model,obj,:midpoint,"Cartpole")
ilqr_dircol_comparison(model,obj,:rk3_foh,"Cartpole")




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                              #
#                            CONSTRAINTS                                       #
#                                                                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Set Solver Options
opts.verbose = true
opts.verbose = false
opts.eps_constraint = 1e-8
opts.eps_intermediate = 1e-6
opts.eps = 1e-8
mesh = [0.1,0.07,0.05]
dt = 0.03

#***************#
#   PENDULUM    #
#***************#

model = Model(Dynamics.pendulum_dynamics!,2,1)
obj0 = Dynamics.pendulum[2]
# model, obj = Dynamics.pendulum!
obj_uncon = copy(obj0)
obj_uncon.tf = 5.
obj_uncon.R .= [1e-1]

u_bnd = 2
obj = ConstrainedObjective(obj_uncon,u_max=u_bnd,u_min=-u_bnd)
name = "Pendulum"
method=:midpoint
time = 0:dt:obj.tf
X0 = -[sin.(time*3) cos.(time*3)]'

ilqr_dircol_comparison(model,obj,:midpoint,"Pendulum (Constrained)",X0,dt=dt,mesh=mesh,opts=opts)
ilqr_dircol_comparison(model,obj,:rk3_foh,"Pendulum (Constrained)",X0,dt=dt,mesh=mesh,opts=opts)


#***************#
#   CART POLE   #
#***************#
model, obj0 = Dynamics.cartpole_analytical
obj = copy(obj0)

obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 30
x_bnd = [1,Inf,Inf,Inf]
dt = 0.01
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd);
method = :rk3

name = "Cartpole"
method = :rk3
dt = 0.01
X0 = []
opts = SolverOptions(verbose=false,
                   eps_constraint=1e-6,
                   eps_intermediate=1e-4,
                   eps=1e-6,
                   outer_loop_update_type=:uniform,
                   constraint_decrease_ratio=0.1)

opts0 = SolverOptions()
name *= "_" * string(method)

# Initial state
N,dt = TrajectoryOptimization.calc_N(obj.tf, dt)
U = ones(1,N)*1
X = zeros(model.n,N)
if isempty(X0)
    X0 = line_trajectory(obj.x0, obj.xf, N)
end

# Feed iLQR to DIRCOL
println("\niLQR → DIRCOL")
solver = Solver(model,obj,dt=dt,integration=method,opts=opts)

println("Solving iLQR...")
solver.opts.verbose = true
time_i1 = @elapsed results, stats_i1 = solve(solver,X0,U)
cost_i1 = cost(solver,results)

println("Solving DIRCOL...")
time_d1 = @elapsed sol_s1, f_opt, stats_d1 = solve_dircol(solver, X0, U, nlp=:snopt, method=:hermite_simpson, grads=:auto, start=:warm)
cost_d1 = cost(solver,sol_s1)

time_d1 = @elapsed sol_i1, f_opt, stats_d1 = solve_dircol(solver, X0, U, nlp=:ipopt, method=:hermite_simpson, grads=:auto, start=:warm)
cost_d1 = cost(solver,sol_i1)



p1 = comparison_plot(results,sol_s1,sol_i1,title=method)
traj_diff = sol_i1.X-results.X
max_err1 = maximum(abs.(traj_diff))
norm_err1 = vecnorm(traj_diff)
