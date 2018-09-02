function comparison_plot(results,X_dircol,U_dircol;kwargs...)
    time_dircol = linspace(0,obj.tf,size(X_dircol,2))
    time_ilqr = linspace(0,obj.tf,size(results.X,2))
    state_labels = ["position (m)" "angle (rad)"]
    colors = [:blue :red]

    p = plot(time_dircol,U_dircol',width=2, label="DIRCOL")
    plot!(time_dircol,results.U',width=2,label="iLQR")
    # display(p)

    p = plot(time_dircol,X_dircol[1:2,:]',width=2, label=state_labels.*" (DIRCOL)",color=colors)
    plot!(time_ilqr,results.X[1:2,:]',width=2,label=state_labels.*" (iLQR)",color=colors, style=:dash,
        xlabel="time (s)", ylabel="States",legend=:topleft;kwargs...)
    display(p)
    return p
end


function ilqr_dircol_comparison(model, obj, method, name)

    name *= "_" * string(method)

    # Set Solver Options
    opts = SolverOptions()
    opts.verbose = false
    opts.eps_constraint = 1e-6
    opts.eps_intermediate = 1e-4
    opts.eps = 1e-8
    mesh = [0.1,0.07,0.05,0.03,0.02] # for dircol
    dt = 0.01

    # Initial state
    N,dt = TrajectoryOptimization.calc_N(obj.tf, dt)
    U = ones(1,N)*1
    X = zeros(model.n,N)
    X0 = line_trajectory(obj.x0, obj.xf, N)

    # Feed iLQR to DIRCOL
    println("\niLQR → DIRCOL")
    solver = Solver(model,obj,dt=dt,integration=method,opts=opts)

    println("Solving iLQR...")
    time_i1 = @elapsed results, stats_i1 = solve(solver,X0,U)
    cost_i1 = cost(solver,results)

    println("Solving DIRCOL...")
    time_d1 = @elapsed x_opt, u_opt, f_opt, stats_d1 = solve_dircol(solver, results.X, results.U, mesh, grads=:none, start=:warm)
    cost_d1 = cost(solver,x_opt,u_opt)

    p1 = comparison_plot(results,x_opt,u_opt,title=method)
    traj_diff = x_opt-results.X
    max_err1 = maximum(abs.(traj_diff))
    norm_err1 = vecnorm(traj_diff)

    # Feed DIRCOL to iLQR
    println("\nDIRCOL → iLQR")
    solver = Solver(model,obj,dt=dt,integration=method,opts=opts)

    println("Solving DIRCOL...")
    time_d2 = @elapsed x_opt2, u_opt2, f_opt, stats_d2 = solve_dircol(solver, X0, U, mesh, grads=:none)
    cost_d2 = cost(solver,x_opt2,u_opt2)

    println("Solving iLQR...")
    time_i2 = @elapsed results2, stats_i2 = solve(solver,x_opt2,u_opt2)
    cost_i2 = cost(solver,results2)

    p2 = comparison_plot(results2,x_opt2,u_opt2,title=method)
    traj_diff = x_opt2-results2.X
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
dt = 0.01
obj.R .= [1e-2]

ilqr_dircol_comparison(model,obj,:midpoint,"Pendulum")
ilqr_dircol_comparison(model,obj,:rk3_foh,"Pendulum")



#***************#
#   CART POLE   #
#***************#
model, obj0 = Dynamics.cartpole_analytical

obj = copy(obj0)
obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 50
x_bnd = [0.6,Inf,Inf,Inf]
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)

ilqr_dircol_comparison(model,obj,:midpoint,"Cartpole")
ilqr_dircol_comparison(model,obj,:rk3_foh,"Cartpole")
