using TrajectoryOptimization
using Base.Test
using Plots

#### Solver setup
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
# opts.c1 = 1e-4
# opts.c2 = 3.0
opts.mu_al_update = 10.0
#opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-6
opts.eps_intermediate = 1e-3
opts.eps = 1e-6
opts.τ = .5
# opts.iterations_outerloop = 100
# opts.iterations = 1000
# opts.iterations_linesearch = 50
###

### System
model,  = TrajectoryOptimization.Dynamics.cartpole_analytical
x0 = [0.;0.;0.;0.]
xf = [0.;pi;0.;0.]

# costs
Q = 0.01*eye(model.n)
Qf = 500.0*eye(model.n)
R = 0.01*eye(model.m)

# simulation
tf = 3.0

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

u_min = -20.0
u_max = 50.0
x_min = [-1.0; -1000; -1000; -1000]
x_max = [1.0; 1000; 1000; 1000]

obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh = Solver(model, obj_con, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh = Solver(model, obj_con, integration=:rk3, dt=dt, opts=opts)

U = ones(solver_foh.model.m,solver_foh.N)
X_interp = line_trajectory(solver_foh)


sol_foh, = TrajectoryOptimization.solve(solver_foh,X_interp,U)
sol_zoh, = TrajectoryOptimization.solve(solver_zoh,X_interp,U)

println("Final state (foh): $(sol_foh.X[:,end])")
println("Final state (zoh): $(sol_zoh.X[:,end])")

println("Termination index\n foh: $(sol_foh.termination_index)\n zoh: $(sol_foh.termination_index)")

println("Final cost (foh): $(sol_foh.cost[sol_foh.termination_index])")
println("Final cost (zoh): $(sol_zoh.cost[sol_zoh.termination_index])")

plot((sol_foh.cost[1:sol_foh.termination_index]))
plot((sol_zoh.cost[1:sol_zoh.termination_index]))

plot(sol_foh.U')
plot!(sol_zoh.U')

plot(sol_foh.X[1:2,:]')
plot!(sol_zoh.X[1:2,:]')

sol_foh
