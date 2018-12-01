# Set random seed
using Random
using Test
Random.seed!(7)

### Solver Options ###
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
######################

### Simple Pendulum ###
obj_uncon_p = TrajectoryOptimization.Dynamics.pendulum![2]
model_p = TrajectoryOptimization.Dynamics.pendulum![1]

## Unconstrained pendulum (foh) ##
solver_foh = TrajectoryOptimization.Solver(model_p, obj_uncon_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(model_p, obj_uncon_p, integration=:rk3, dt=dt, opts=opts)

U = ones(solver_foh.model.m, solver_foh.N)

sol_zoh, = TrajectoryOptimization.solve(solver_zoh,U)
sol_foh, = TrajectoryOptimization.solve(solver_foh,U)

# Test final state from foh solve
@test norm(solver_foh.obj.xf - sol_foh.X[end]) < 1e-3
##################################

## Control constraints pendulum (foh) ##
u_min = -2.0
u_max = 2.0
obj_con_p = TrajectoryOptimization.ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max) # constrained objective

solver_foh_con = Solver(model_p, obj_con_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con = Solver(model_p, obj_con_p, integration=:rk3, dt=dt, opts=opts)

U = ones(solver_foh_con.model.m, solver_foh_con.N)

sol_foh_con, = TrajectoryOptimization.solve(solver_foh_con,U)
sol_zoh_con, = TrajectoryOptimization.solve(solver_zoh_con,U)

# Test final state from foh solve
@test norm(sol_foh_con.X[end] - solver_foh_con.obj.xf) < 1e-3
########################################

## State and control constraints pendulum (foh) ##
u_min = -20
u_max = 6
x_min = [-10; -2]
x_max = [10; 6]
obj_con2_p = TrajectoryOptimization.ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(model_p, obj_con2_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con2 = Solver(model_p, obj_con2_p, integration=:rk3, dt=dt, opts=opts)

sol_foh_con2, = TrajectoryOptimization.solve(solver_foh_con2,U)
sol_zoh_con2, = TrajectoryOptimization.solve(solver_zoh_con2,U)

# Test final state from foh solve
@test norm(sol_foh_con2.X[end] - solver_foh_con2.obj.xf) < 1e-3
###################################################

## Unconstrained infeasible start pendulum (foh) ##
solver_uncon_inf = Solver(model_p,obj_uncon_p,integration=:rk3_foh,dt=dt,opts=opts)

# -initial state and control trajectories
X_interp = line_trajectory(solver_uncon_inf.obj.x0,solver_uncon_inf.obj.xf,solver_uncon_inf.N)
U = ones(solver_uncon_inf.model.m, solver_uncon_inf.N)

results_inf, = solve(solver_uncon_inf,X_interp,U)

# Test final (dynamically constrained) state from foh solve
@test norm(results_inf.X[end] - solver_uncon_inf.obj.xf) < 1e-3

####################################################

## Infeasible start with constraints pendulum (foh) ##
u_min = -3
u_max = 3
x_min = [-10;-10]
x_max = [10; 10]

obj_con_p = ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)
opts.verbose = false
solver_con2 = Solver(model_p,obj_con_p,integration=:rk3_foh,dt=dt,opts=opts)

# -Linear interpolation for state trajectory
X_interp = line_trajectory(solver_con2.obj.x0,solver_con2.obj.xf,solver_con2.N)
U = ones(solver_con2.model.m,solver_con2.N)

results_inf2, stats_inf2 = solve(solver_con2,X_interp,U)

# plot(to_array(results_inf2.X)')
# plot(stats_inf2["cost"][1:15])
# stats_inf2["cost"][end]

# Test final state from foh solve
@test norm(results_inf2.X[end] - solver_con2.obj.xf) < 1e-3
######################################################

#----------------#

### Dubins Car ###
obj_uncon_dc = TrajectoryOptimization.Dynamics.dubinscar![2]
model_dc = Dynamics.dubinscar![1]

## Unconstrained Dubins car (foh) ##
solver_foh = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3_foh,opts=opts)
solver_zoh = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3,opts=opts)

U = rand(solver_foh.model.m, solver_foh.N)
sol_zoh, = TrajectoryOptimization.solve(solver_zoh,U)
sol_foh, = TrajectoryOptimization.solve(solver_foh,U)

# Test final state from foh solve
@test norm(sol_foh.X[end] - solver_foh.obj.xf) < 1e-3

#####################################

## State and control constraints Dubins car (foh) ##
u_min = [-1; -1]
u_max = [100; 100]
x_min = [0; -100; -100]
x_max = [1.0; 100; 100]

obj_con2_dc = TrajectoryOptimization.ConstrainedObjective(obj_uncon_dc, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3, dt=dt, opts=opts)

U = 5*rand(solver_foh_con2.model.m,solver_foh_con2.N)

sol_foh_con2, = TrajectoryOptimization.solve(solver_foh_con2,U)
sol_zoh_con2, = TrajectoryOptimization.solve(solver_zoh_con2,U)

# Test final state from foh solve
@test norm(sol_foh_con2.X[end] - solver_foh_con2.obj.xf) < 1e-3
####################################################

## Infeasible start with state and control constraints Dubins car (foh) ##
u_min = [-1; -1]
u_max = [100; 100]
x_min = [0; -100; -100]
x_max = [1.0; 100; 100]

obj_con2_dc = TrajectoryOptimization.ConstrainedObjective(obj_uncon_dc, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3, dt=dt, opts=opts)

# -initial control and state trajectories
U = 5*rand(solver_foh_con2.model.m,solver_foh_con2.N)
X_interp = line_trajectory(solver_foh_con2)

sol_foh_con2, = TrajectoryOptimization.solve(solver_foh_con2,X_interp,U)
sol_zoh_con2, = TrajectoryOptimization.solve(solver_zoh_con2,X_interp,U)

sol_foh_con2.X[end]
@test norm(sol_foh_con2.X[end] - solver_foh_con2.obj.xf) < 1e-3
sol_foh_con2.X[end]
###########################################################################
