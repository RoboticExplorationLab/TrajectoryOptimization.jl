using TrajectoryOptimization
using Base.Test
using Plots

### Solver Options ###
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
# opts.c1 = 1e-4
# opts.c2 = 3.0
# opts.mu_al_update = 10.0
#opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps_intermediate = 1e-3
opts.eps = 1e-3
opts.τ = 0.1
opts.outer_loop_update = :uniform
# opts.iterations_outerloop = 100
# opts.iterations = 1000
# opts.iterations_linesearch = 50
######################

### Set up model, objective, solver ###
# Model
dt = 0.01
model,  = TrajectoryOptimization.Dynamics.cartpole_analytical

# Objective
Q = 0.01*eye(model.n)
Qf = 10000.0*eye(model.n)
R = 0.0001*eye(model.m)

x0 = [0.;0.;0.;0.]
xf = [0.;pi;0.;0.]

tf = 5.0

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

# -Constraints
u_min = -20
u_max = 1000
x_min = [-1000; -1000; -1000; -1000]
x_max = [1000; 1000; 1000; 1000]

obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

# Solver (foh & zoh)
# solver_foh = Solver(model, obj_con, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh = Solver(model, obj_con, integration=:rk3, dt=dt, opts=opts)

# -Initial control and state trajectories
U = ones(solver_zoh.model.m,solver_zoh.N)
X_interp = line_trajectory(solver_zoh)
# X_interp = ones(solver_zoh.model.n,solver.N)
#######################################

### Solve ###
# @time sol_foh, = TrajectoryOptimization.solve(solver_foh,X_interp,U)
@time sol_zoh, = TrajectoryOptimization.solve(solver_zoh,U)
#############

# ### Results ###
# if opts.verbose
#     # println("Final state (foh): $(sol_foh.X[:,end])")
#     println("Final state (zoh): $(sol_zoh.X[:,end])")
#
#     # println("Termination index\n foh: $(sol_foh.termination_index)\n zoh: $(sol_foh.termination_index)")
#
#     # println("Final cost (foh): $(sol_foh.cost[sol_foh.termination_index])")
#     println("Final cost (zoh): $(sol_zoh.cost[sol_zoh.termination_index])")
#
#     # plot((sol_foh.cost[1:sol_foh.termination_index]))
#     plot!((sol_zoh.cost[1:sol_zoh.termination_index]))
#
#     # plot(sol_foh.U')
#     plot!(sol_zoh.U')
#
#     # plot(sol_foh.X[1:2,:]')
#     plot!(sol_zoh.X[1:2,:]')
# end
###############
