using TrajectoryOptimization.Dynamics
using Base.Test

# Set up models and objective
u_bound = 2.
model,obj = TrajectoryOptimization.Dynamics.pendulum
obj.tf = 5.
model! = TrajectoryOptimization.Model(Dynamics.pendulum_dynamics!,2,1) # inplace dynamics
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective


### UNCONSTRAINED ###
# rk4
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1)
U = ones(solver.model.m, solver.N-1)
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

#  with square root
solver.opts.square_root = true
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3


# midpoint
solver = TrajectoryOptimization.Solver(model,obj,integration=:midpoint,dt=0.1)
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

#  with square root
solver.opts.square_root = true
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3


### CONSTRAINED ###
# rk4
opts = TrajectoryOptimization.SolverOptions()
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts)
results_c = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

#   with Square Root
solver.opts.square_root = true
results_c = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2


# midpoint
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1)
results_c = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

#   with Square Root
solver.opts.square_root = true
results_c = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2



### In-place dynamics ###
# Unconstrained
opts = TrajectoryOptimization.SolverOptions()
solver = TrajectoryOptimization.Solver(model!,obj,dt=0.1,opts=opts)
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

# Constrained
solver = TrajectoryOptimization.Solver(model!,obj_c,dt=0.1,opts=opts)
results_c = TrajectoryOptimization.solve(solver,U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

# Constrained - midpoint
solver = TrajectoryOptimization.Solver(model!,obj_c, integration=:midpoint, dt=0.1, opts=opts)
results_c = TrajectoryOptimization.solve(solver,U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
