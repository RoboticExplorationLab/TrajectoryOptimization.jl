using TrajectoryOptimization.Dynamics
using Base.Test

# Set up models and objective
model,obj = Dynamics.pendulum
model! = TrajectoryOptimization.Model(Dynamics.pendulum_dynamics!,2,1) # inplace dynamics
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-2, u_max=2) # constrained objective

### UNCONSTRAINED ###
# rk4
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1)
U = ones(solver.model.m, solver.N-1)
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

#  with square root
solver.opts.square_root = true
results = solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3


# midpoint
solver = Solver(model,obj,integration=:midpoint,dt=0.1)
results = solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

#  with square root
solver.opts.square_root = true
results = solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3


### CONSTRAINED ###
# rk4
solver = Solver(model,obj_c,dt=0.1)
results_c = solve(solver, U)
max_c = max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

#   with Square Root
solver.opts.square_root = true
results_c = solve(solver, U)
max_c = max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2


# midpoint
solver = Solver(model,obj_c,dt=0.1)
results_c = solve(solver, U)
max_c = max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

#   with Square Root
solver.opts.square_root = true
results_c = solve(solver, U)
max_c = max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2



### In-place dynamics ###
# Unconstrained
opts = SolverOptions()
solver = Solver(model!,obj,dt=0.1,opts=opts)
results = solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

# Constrained
solver = Solver(model!,obj_c,dt=0.1,opts=opts)
results = solve(solver,U)
max_c = max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

# Constrained - midpoint
solver = Solver(model!,obj_c, integration=:midpoint, dt=0.1, opts=opts)
results = solve(solver,U)
max_c = max_violation(results_c)
@test norm(results.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
