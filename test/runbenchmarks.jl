using BenchmarkTools
using LinearAlgebra
using Test

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
solver.opts.use_static = false
results0, = TrajectoryOptimization.solve(solver,U)
norm(results0.X - to_array(results.X),Inf)
norm(to_array(results0.X) - to_array(results.X),Inf)

err = norm(results.X[end]-obj.xf)
err = norm(results0.X[:,end]-obj.xf)
@test err < 1e-3
println("$system - Unconstrained")
solver.opts.use_static = false
@btime TrajectoryOptimization.solve(solver,U)


# Constrained
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts,integration=integration)
solver.opts.verbose = false
solver.opts.cost_intermediate_tolerance = 1e-2
solver.opts.use_static = true
results_c,stats = TrajectoryOptimization.solve(solver,U)
solver.opts.use_static = false
results_c0,stats0 = TrajectoryOptimization.solve(solver,U)

max_c = TrajectoryOptimization.max_violation(results_c)
max_c = TrajectoryOptimization.max_violation(results_c0)
err = norm(results_c.X[end]-obj.xf)
err = norm(results_c0.X[:,end]-obj.xf)
norm(to_array(results_c.X) - results_c0.X, Inf)
cost(solver,results_c)
cost(solver,results_c0)

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
solver.opts.use_static = false
results_inf0, stats0 = TrajectoryOptimization.solve(solver,X_interp,U)

max_c = TrajectoryOptimization.max_violation(results_inf)
max_c = TrajectoryOptimization.max_violation(results_inf0)
err = norm(results_inf.X[end]-obj.xf)
err = norm(results_inf0.X[:,end]-obj.xf)
maximum(to_array(results_inf.X) - results_inf0.X)

@test norm(results_inf.X[end]-obj.xf) < 1e-3
@test max_c < 1e-2
println("$system - Infeasible Start")
@btime TrajectoryOptimization.solve(solver,U)
