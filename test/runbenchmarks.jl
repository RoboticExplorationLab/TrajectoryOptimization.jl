using BenchmarkTools
using LinearAlgebra
using Test
using TrajectoryOptimization: to_array
using Logging

println("\nRUNNING BENCHMARKS\n")
system = "Simple Pendulum"

# Set up models and objective
u_bound = 2.
model,obj = TrajectoryOptimization.Dynamics.pendulum!
opts = TrajectoryOptimization.SolverOptions()
integration = :rk3

obj.Q .= Diagonal(I,2)*1e-3
obj.R .= Diagonal(I,1)*1e-3
obj.Qf .= Diagonal(I,2)*30
obj.tf = 5.
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective
obj = TrajectoryOptimization.to_static(obj)
obj_c =TrajectoryOptimization.to_static(obj_c)

# Unconstrained
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts,integration=integration)
solver.opts.verbose = false
U = ones(model.m,solver.N)
solver.opts.use_static = true
results, = TrajectoryOptimization.solve(solver,U)

err = norm(results.X[end]-obj.xf)
@test err < 1e-3
println("$system - Unconstrained")
disable_logging(Logging.Info)
@btime TrajectoryOptimization.solve(solver,U)


# Constrained
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts,integration=integration)
solver.opts.verbose = false
solver.opts.cost_intermediate_tolerance = 1e-2
solver.opts.use_static = true
results_c,stats = TrajectoryOptimization.solve(solver,U)

max_c = TrajectoryOptimization.max_violation(results_c)
err = norm(results_c.X[end]-obj.xf)

@test norm(results_c.X[end]-obj.xf) < 1e-3
@test max_c < 1e-2
println("$system - Constrained")
@btime TrajectoryOptimization.solve(solver,U)

### Infeasible Start
obj_c2 = TrajectoryOptimization.update_objective(obj_c)
solver = TrajectoryOptimization.Solver(model, obj_c2, dt=0.1, opts=opts, integration=integration)
X_interp = TrajectoryOptimization.line_trajectory(obj.x0, obj.xf,solver.N)
solver.opts.verbose = false
solver.opts.solve_feasible = false
solver.opts.use_static = true
results_inf, stats = TrajectoryOptimization.solve(solver,X_interp,U)

max_c = TrajectoryOptimization.max_violation(results_inf)
err = norm(results_inf.X[end]-obj.xf)

@test norm(results_inf.X[end]-obj.xf) < 1e-3
@test max_c < 1e-2
println("$system - Infeasible Start")
@btime TrajectoryOptimization.solve(solver,U)
