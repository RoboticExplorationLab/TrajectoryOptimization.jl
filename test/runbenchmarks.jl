using BenchmarkTools
using LinearAlgebra

println("\nRUNNING BENCHMARKS\n")
system = "Simple Pendulum"

# Set up models and objective
u_bound = 2.
model,obj = TrajectoryOptimization.Dynamics.pendulum!
opts = TrajectoryOptimization.SolverOptions()

obj.Q .= Diagonal(I,2)*1e-3
obj.R .= Diagonal(I,1)*1e-3
obj.Qf .= Diagonal(I,2)*30
obj.tf = 5.
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective

# Unconstrained
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
U = ones(model.m,solver.N)
@time results,stats = TrajectoryOptimization.solve(solver,U) # Test random init
err = norm(results.X[:,end]-obj.xf)
@test err < 1e-3
println("$system - Unconstrained")
@btime TrajectoryOptimization.solve(solver,U)

# Constrained
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts)
@time results_c, = TrajectoryOptimization.solve(solver,U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
println("$system - Constrained")
@btime TrajectoryOptimization.solve(solver,U)

### Infeasible Start
obj_c2 = TrajectoryOptimization.update_objective(obj_c)
solver = TrajectoryOptimization.Solver(model, obj_c2, dt=0.1, opts=opts)
X_interp = TrajectoryOptimization.line_trajectory(obj.x0, obj.xf,solver.N)
results_inf, = TrajectoryOptimization.solve(solver,X_interp,U)
max_c = TrajectoryOptimization.max_violation(results_inf)
@test norm(results_inf.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2
println("$system - Infeasible Start")
@btime TrajectoryOptimization.solve(solver,U)
