using TrajectoryOptimization
using Base.Test
using Juno

# Set up models and objective
u_bound = 2.
model,obj = TrajectoryOptimization.Dynamics.pendulum
opts = TrajectoryOptimization.SolverOptions()
opts.c1 = 1e-3
opts.c2 = 1.5
opts.mu_al_update = 100.

obj.Q .= eye(2)*1e-3
obj.R .= eye(1)*1e-3
obj.tf = 5.
model! = TrajectoryOptimization.Model(Dynamics.pendulum_dynamics!,2,1) # inplace dynamics
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective


### UNCONSTRAINED ###
# rk4
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
U = zeros(solver.model.m, solver.N-1)
# @enter TrajectoryOptimization.solve(solver,U)
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

#  with square root
solver.opts.square_root = true
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3


# midpoint
solver = TrajectoryOptimization.Solver(model,obj,integration=:midpoint,dt=0.1,opts=opts)
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3

#  with square root
solver.opts.square_root = true
results = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[:,end]-obj.xf) < 1e-3


### CONSTRAINED ###
# rk4
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
results = TrajectoryOptimization.solve(solver) # Test random init
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

### Infeasible Start
opts.cache = true
opts.verbose = true
obj_c2 = TrajectoryOptimization.update_objective(obj_c, u_min=-Inf, x_min=[-5;-5], x_max=[10;10])
solver = TrajectoryOptimization.Solver(model!, obj_c2, dt=0.1, opts=opts)
X_interp = TrajectoryOptimization.line_trajectory(obj.x0, obj.xf,solver.N)
results_inf = TrajectoryOptimization.solve_al(solver,X_interp,U)
max_c = TrajectoryOptimization.max_violation(results_inf.result[end])
@test norm(results_inf.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
@test minimum(any(results_inf.U' .< -obj_c2.u_max[1])) # Make sure lower bound is unbounded

### OTHER TESTS ###
# Test undefined integration
@test_throws ArgumentError TrajectoryOptimization.Solver(model!,obj_c, integration=:bogus, dt=0.1, opts=opts)
