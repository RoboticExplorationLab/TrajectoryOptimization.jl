reload("TrajectoryOptimization")
using TrajectoryOptimization
using Base.Test
using BenchmarkTools

println("\nRUNNING BENCHMARKS\n")
system = "Simple Pendulum"

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

# Unconstrained
solver = TrajectoryOptimization.Solver(model!,obj,dt=0.1,opts=opts)
U = ones(model!.m,solver.N-1)
results = TrajectoryOptimization.solve_al(solver,U) # Test random init
err = norm(results.X[:,end]-obj.xf)
@test err < 1e-3
println("$system - Unconstrained")
@btime TrajectoryOptimization.solve(solver,U)

# Constrained
solver = TrajectoryOptimization.Solver(model!,obj_c,dt=0.1,opts=opts)
results_c = TrajectoryOptimization.solve(solver,U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
println("$system - Constrained")
@btime TrajectoryOptimization.solve(solver,U)
@profiler TrajectoryOptimization.solve(solver,U)

### Infeasible Start
obj_c2 = TrajectoryOptimization.update_objective(obj_c)
solver = TrajectoryOptimization.Solver(model!, obj_c2, dt=0.1, opts=opts)
X_interp = TrajectoryOptimization.line_trajectory(obj.x0, obj.xf,solver.N)
results_inf = TrajectoryOptimization.solve_al(solver,X_interp,U)
max_c = TrajectoryOptimization.max_violation(results_inf)
@test norm(results_inf.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
println("$system - Infeasible Start")
@btime TrajectoryOptimization.solve(solver,U)
