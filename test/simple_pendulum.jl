using TrajectoryOptimization
using Base.Test
#using Juno

# Set up models and objective
u_bound = 3.
model,obj = TrajectoryOptimization.Dynamics.pendulum
opts = TrajectoryOptimization.SolverOptions()
opts.c1 = 1e-3
opts.c2 = 2.0
opts.mu_al_update = 100.

obj.Q .= eye(2)*1e-3
obj.R .= eye(1)*1e-2
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
results_sr = TrajectoryOptimization.solve(solver,U)
@test norm(results_sr.X[:,end]-obj.xf) < 1e-3
@test norm(results_sr.X - results.X) ≈ 0. atol=1e-12

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
opts.verbose = false
obj_c2 = TrajectoryOptimization.update_objective(obj_c, u_min=-Inf, u_max=1, x_min=[-5;-5], x_max=[10;10])
solver = TrajectoryOptimization.Solver(model!, obj_c2, dt=0.1, opts=opts)
X_interp = TrajectoryOptimization.line_trajectory(obj.x0, obj.xf,solver.N)
results_inf = TrajectoryOptimization.solve(solver,X_interp,U)
max_c = TrajectoryOptimization.max_violation(results_inf.result[end])
@test norm(results_inf.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
@test abs(minimum(results_inf.U) + 2) > 0.5 # Make sure lower bound is unbounded

# test linear interpolation for state trajectory
@test norm(X_interp[:,1] - solver.obj.x0) < 1e-8
@test norm(X_interp[:,end] - solver.obj.xf) < 1e-8
@test all(X_interp[1,2:end-1] .<= max(solver.obj.x0[1],solver.obj.xf[1]))
@test all(X_interp[2,2:end-1] .<= max(solver.obj.x0[2],solver.obj.xf[2]))

# test that additional augmented controls can achieve an infeasible state trajectory
U_infeasible = ones(solver.model.m,solver.N-1)
X_infeasible = ones(solver.model.n,solver.N)
solver.obj.x0 = ones(solver.model.n)
ui = TrajectoryOptimization.infeasible_controls(solver,X_infeasible,U_infeasible)
results_infeasible = TrajectoryOptimization.ConstrainedResults(solver.model.n,solver.model.m+solver.model.n,1,solver.N,1)
results_infeasible.U[:,:] = [U_infeasible;ui]
# solver.opts.infeasible = true  # solver needs to know to use an infeasible rollout
TrajectoryOptimization.rollout!(results_infeasible,solver)

@test all(ui[1,:] .== ui[1,1]) # special case for state trajectory of all ones, control 1 should all be same
@test all(ui[2,:] .== ui[2,1]) # special case for state trajectory of all ones, control 2 should all be same
@test all(results_infeasible.X .== X_infeasible)
# rolled out trajectory should be equivalent to infeasible trajectory after applying augmented controls


### OTHER TESTS ###
# Test undefined integration
@test_throws ArgumentError TrajectoryOptimization.Solver(model!,obj_c, integration=:bogus, dt=0.1, opts=opts)
