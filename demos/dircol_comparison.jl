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
end

# Cart Pole
state = Dynamics.cartpole_mechanism
model, obj = Dynamics.cartpole_analytical

obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 50
x_bnd = [0.6,Inf,Inf,Inf]
dt = 0.01
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)
objective = copy(obj)
mesh = [0.1,0.07,0.05,0.03]

# Initial State
N,dt = TrajectoryOptimization.calc_N(obj.tf, dt)
U = ones(1,N)*1
X = zeros(model.n,N)
X0 = line_trajectory(obj.x0, obj.xf, N)

#***************#
#   MIDPOINT    #
#***************#

# Feed DIRCOL to iLQR
solver = Solver(model,objective,dt=dt,integration=:midpoint)
solver.opts.verbose = true
rollout!(X,U,solver)

x_opt, u_opt, f_opt = solve_dircol(solver, X, U, mesh, method=:midpoint, grads=:none)
@time results = solve(solver,u_opt)
comparison_plot(results,x_opt,u_opt,title="Midpoint ZOH")

# Feed iLQR to DIRCOL
solver = Solver(model,objective,dt=dt,integration=:midpoint)
solver.opts.verbose = true
rollout!(X,U,solver)
@time results = solve(solver,X0,U)

x_opt, u_opt, f_opt = solve_dircol(solver, results.X, results.U, mesh, method=:midpoint, grads=:none, start=:warm)
solver.opts.infeasible = false
comparison_plot(results,x_opt,u_opt,title="Midpoint ZOH")



#***************#
#    RK3 FOH    #
#***************#

# Feed DIRCOL to iLQR
solver = Solver(model,objective,dt=dt,integration=:rk3_foh)
solver.opts.verbose = true
rollout!(X,U,solver)

x_opt, u_opt, f_opt = solve_dircol(solver, X, U, mesh, grads=:none)
@time results = solve(solver,u_opt)
comparison_plot(results,x_opt,u_opt,title="RK3 FOH")

# Feed iLQR to DIRCOL
solver = Solver(model,objective,dt=dt,integration=:midpoint)
solver.opts.verbose = true
rollout!(X,U,solver)
@time results = solve(solver,X0,U)

x_opt, u_opt, f_opt = solve_dircol(solver, results.X, results.U, mesh, grads=:none, start=:warm)
solver.opts.infeasible = false
comparison_plot(results,x_opt,u_opt,title="RK3 FOH")

cost(solver,results)
cost(solver,x_opt,u_opt)

#***************#
#   PENDULUM    #
#***************#
model = Model(Dynamics.pendulum_dynamics!,2,1)
obj = Dynamics.pendulum[2]
# model, obj = Dynamics.pendulum!
obj = copy(obj)
obj.tf = 5.
dt = 0.01
# obj.R .*= 10
mesh = [0.1,0.07,0.05,0.03,0.02]


# Initial state
N,dt = TrajectoryOptimization.calc_N(obj.tf, dt)
U = ones(1,N)*1
X = zeros(model.n,N)
X0 = line_trajectory(obj.x0, obj.xf, N)

# Feed iLQR to DIRCOL
solver = Solver(model,obj,dt=dt,integration=:rk4)
solver.opts.verbose = true
rollout!(X,U,solver)
@time results = solve(solver,X0,U)
cost(solver,results)

x_opt, u_opt, f_opt = solve_dircol(solver, results.X, results.U, grads=:none, start=:warm)
cost(solver,x_opt,u_opt)
solver.opts.infeasible = false
comparison_plot(results,x_opt,u_opt,title="RK3 FOH")


# Feed DIRCOL to iLQR
solver = Solver(model,obj,dt=dt,integration=:rk4)
solver.opts.verbose = true
solver.opts.eps_constraint = 1e-5
rollout!(X,U,solver)

x_opt2, u_opt2, f_opt = solve_dircol(solver, X, U, mesh, grads=:none)
cost(solver,x_opt2,u_opt2)
rollout!(X,u_opt2,solver)
plot(x_opt2')
plot!(X')
@time results2 = solve(solver,x_opt2,u_opt2)
comparison_plot(results2,x_opt2,u_opt2,title="RK3 FOH")

ui = TrajectoryOptimization.infeasible_controls(solver,x_opt2,u_opt2)
u_aug = [u_opt2;ui]
x_aug = zeros(x_opt2)
rollout!(x_aug,u_aug,solver)
plot!(x_aug')
for i = 1:100
    @time results2 = solve(solver,results2.X,results2.U)
    comparison_plot(results2,x_opt2,u_opt2,title="RK3 FOH")
end
