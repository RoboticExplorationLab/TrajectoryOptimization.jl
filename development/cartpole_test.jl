### Solver Options ###
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = true
######################

### Set up model, objective, solver ###
# Model
dt = 0.1
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf_cartpole = joinpath(urdf_folder, "cartpole.urdf")

model_urdf = Model(urdf_cartpole,[1.;0.])
n = model.n
m = model.m
# Objective
Q = 0.01*Matrix(I,n,n)
Qf = 1000.0*Matrix(I,n,n)
R = 0.01*Matrix(I,m,m)

x0 = [0.;pi;0.;0.]
xf = [0.;0.;0.;0.]

tf = 5.0

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

# -Constraints
u_min = -20
u_max = 1000
x_min = [-1000; -1000; -1000; -1000]
x_max = [1000; 1000; 1000; 1000]

obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

# Solver (foh & zoh)
# solver_foh = Solver(model, obj_con, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh = Solver(model, obj_uncon, integration=:rk3, dt=dt, opts=opts)

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
