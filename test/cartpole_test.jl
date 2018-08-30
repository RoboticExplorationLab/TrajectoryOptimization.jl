using TrajectoryOptimization
using Base.Test
using Plots

srand(1)
#### Solver setup
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
opts.c1 = 1e-4
opts.c2 = 10.0
opts.mu_al_update = 10.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-5
opts.iterations_outerloop = 100
opts.iterations = 1000
opts.iterations_linesearch = 50
###

### System
model, obj_uncon = TrajectoryOptimization.Dynamics.cartpole_analytical
u_min = -100
u_max = 100
x_min = [-100; -100; -100; -100]
x_max = [100; 100; 100; 100]

obj_uncon.R[:] = (1e-2)*eye(1)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver = Solver(model, obj_con, integration=:rk3_foh, dt=dt, opts=opts)

U = zeros(solver.model.m,solver.N)
X_interp = line_trajectory(solver)


sol = TrajectoryOptimization.solve(solver,X_interp,U)
plot(sol.X')

println("Final state (foh): $(sol.X[:,end])")

println("Final cost (foh): $(sol.cost[sol.termination_index])")

plot(sol.cost[1:sol.termination_index])
plot(sol.U')
