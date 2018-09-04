using Plots
function comparison_plot(results,X_dircol,U_dircol;kwargs...)
    n = size(results.X,1)
    if n == 2 # pendulum
        state_labels = ["position (rad)" "vel (rad/s)"]
    else
        state_labels = ["position (m)" "angle (rad)"]
    end
    time_dircol = linspace(0,obj.tf,size(X_dircol,2))
    time_ilqr = linspace(0,obj.tf,size(results.X,2))

    colors = [:blue :red]

    p_u = plot(time_dircol,U_dircol',width=2, label="DIRCOL")
    plot!(time_dircol,results.U',width=2,label="iLQR")
    display(p_u)

    p = plot(time_dircol,X_dircol[1:2,:]',width=2, label=state_labels.*" (DIRCOL)",color=colors)
    plot!(time_ilqr,results.X[1:2,:]',width=2,label=state_labels.*" (iLQR)",color=colors, style=:dash,
        xlabel="time (s)", ylabel="States",legend=:topleft;kwargs...)
    display(p)
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
    time_d1 = @elapsed x_opt, u_opt, f_opt, stats_d1 = solve_dircol(solver, results.X, results.U, grads=:none, start=:warm)
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
x_bnd = [0.55,Inf,Inf,Inf]
dt = 0.01
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd);
method = :rk3

solver = Solver(model,obj_con,dt=dt)
U = ones(1,solver.N)*10
X = line_trajectory(obj.x0, obj.xf, solver.N)
solver.opts.verbose = true
@time results, = solve(solver,X,U);
cost(solver,results)
plot(results.X')
plot(results.U')
plot!(results.LAMBDA')

# Initial state
N,dt = TrajectoryOptimization.calc_N(obj.tf, dt)

U = ones(1,N)*10
X = zeros(model.n,N)
# X0 = -[sin.(time*3) cos.(time*3)]'
X0 = line_trajectory(obj.x0,obj.xf,N)

opts = SolverOptions()
solver = Solver(model,obj,dt=dt,integration=method,opts=opts)
rollout!(X,U,solver)
solver.opts.verbose = true
solver.opts.solve_feasible = true
time_i1 = @elapsed results, stats = solve(solver,U)

plot(results.X[1:2,:]')
plot(results.U')
cost_i1 = cost(solver,results)


println("Solving DIRCOL...")
solver.opts.verbose = true
time_d1 = @elapsed x_opt, u_opt, f_opt, stats_d1 = solve_dircol(solver, results.X, results.U, grads=:none, start=:warm)
cost_d1 = cost(solver,x_opt,u_opt)

p1 = comparison_plot(results,x_opt,u_opt,title=method)
traj_diff = x_opt-results.X
max_err1 = maximum(abs.(traj_diff))
norm_err1 = vecnorm(traj_diff)

solver = Solver(model,obj,dt=dt,integration=method,opts=opts)
solver.opts.infeasible_regularization = 1e6
solver.opts.solve_feasible = true
solver.opts.cache = true
time_i2 = @elapsed results2, stats_i2 = solve(solver,x_opt,u_opt)
cost_i2 = cost(solver,results2.X, results2.U)
r1 = copy(results2.result[1])


solver.opts.infeasible_regularization = 1e6
getR(solver)
calc_jacobians(r1, solver)
v1,v2 = backwardpass!(r1,solver)
J = forwardpass!(r1,solver,v1,v2)
plot(r1.X')
plot!(r1.X_')
plot(r1.U[:,:]')
plot!(r1.U_[:,:]')



p2 = comparison_plot(results2,x_opt,u_opt,title=method)

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
