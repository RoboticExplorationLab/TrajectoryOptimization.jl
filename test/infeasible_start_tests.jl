### Solver options ###
opts = SolverOptions()
opts.square_root = false
opts.verbose=false
opts.constraint_tolerance = 1e-5
opts.cost_tolerance = 1e-6

model, obj = TrajectoryOptimization.Dynamics.pendulum
solver = Solver(model,obj,dt=0.1,opts=opts)

# -Initial state and control trajectories
X_interp = line_trajectory(solver)
U = ones(solver.model.m,solver.N-1)

results, stats = solve(solver,X_interp,U)

@test norm(results.X[end] - solver.obj.xf) < 1e-5
@test max_violation(results) < 1e-5
@test 0.5*(stats["iterations"] - stats["iterations (infeasible)"]) < stats["iterations (infeasible)"]
@test 0.5*(stats["outer loop iterations"] - stats["outer loop iterations (infeasible)"]) < stats["outer loop iterations (infeasible)"]

# Constraints
u_min = -2
u_max = 2
x_min = [-10;-10]
x_max = [10; 10]
obj_c = ConstrainedObjective(obj, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

solver_con = Solver(model,obj_c,dt=0.1,opts=opts)

# Linear interpolation for state trajectory
X_interp = line_trajectory(solver_con)
U = ones(solver_con.model.m,solver_con.N-1)

results_con, stats_con = solve(solver_con,X_interp,U)
@test norm(results_con.X[end] - solver_con.obj.xf) < 1e-5
@test max_violation(results_con) < 1e-5
@test 0.5*(stats_con["iterations"] - stats_con["iterations (infeasible)"]) < stats_con["iterations (infeasible)"]
@test 0.5*(stats_con["outer loop iterations"] - stats_con["outer loop iterations (infeasible)"]) < stats_con["outer loop iterations (infeasible)"]
