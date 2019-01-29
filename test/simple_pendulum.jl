# Set up models and objective
model, obj = TrajectoryOptimization.Dynamics.pendulum
obj_c = Dynamics.pendulum_constrained[2]
opts = TrajectoryOptimization.SolverOptions()
opts.cost_tolerance = 1e-5
opts.constraint_tolerance = 1e-5


### UNCONSTRAINED ###
# rk4
solver = TrajectoryOptimization.Solver(model,obj,dt=0.01,opts=opts)
U = zeros(solver.model.m, solver.N-1)
results, = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[end]-obj.xf) < 1e-3

# midpoint
solver = TrajectoryOptimization.Solver(model,obj,integration=:midpoint,dt=0.1,opts=opts)
results, = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[end]-obj.xf) < 1e-3

# random control initialization
solver.opts
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1)
results, =TrajectoryOptimization.solve(solver) # Test random init
@test norm(results.X[end]-obj.xf) < 1e-3

### CONSTRAINED ###
# rk4
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts)
results_c, = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[end]-obj.xf) < 1e-5
@test max_c < 1e-5

# midpoint
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts)
results_c, = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[end]-obj.xf) < 1e-5
@test max_c < 1e-5

# Constrained
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts)
results_c, = TrajectoryOptimization.solve(solver,U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[end]-obj.xf) < 1e-5
@test max_c < 1e-5

# Constrained - midpoint
solver = TrajectoryOptimization.Solver(model,obj_c, integration=:midpoint, dt=0.1, opts=opts)
results_c, = TrajectoryOptimization.solve(solver,U)
max_c = TrajectoryOptimization.max_violation(results_c)
@test norm(results_c.X[end]-obj.xf) < 1e-5
@test max_c < 1e-5

# Infeasible Start
solver_inf = TrajectoryOptimization.Solver(model, obj_c, dt=0.1, opts=opts)
X_interp = TrajectoryOptimization.line_trajectory(solver_inf)
results_inf, = TrajectoryOptimization.solve(solver_inf,X_interp,U)
max_c = TrajectoryOptimization.max_violation(results_inf)
@test norm(results_inf.X[end]-obj.xf) < 1e-5
@test max_c < 1e-5

# test linear interpolation for state trajectory
@test norm(X_interp[:,1] - solver_inf.obj.x0) < 1e-8
@test norm(X_interp[:,end] - solver.obj.xf) < 1e-8
@test all(X_interp[1,2:end-1] .<= max(solver.obj.x0[1],solver.obj.xf[1]))
@test all(X_interp[2,2:end-1] .<= max(solver.obj.x0[2],solver.obj.xf[2]))

# test that additional augmented controls can achieve an infeasible state trajectory
obj_c_2 = copy(obj_c)
obj_c_2.x0[:] = ones(solver.model.n)
solver = TrajectoryOptimization.Solver(model, obj_c_2, dt=0.1, opts=opts)
U_infeasible = ones(solver.model.m,solver.N-1)
X_infeasible = ones(solver.model.n,solver.N)
solver.state.infeasible = true  # solver needs to know to use an infeasible rollout
p, pI, pE = TrajectoryOptimization.get_num_constraints(solver::Solver)
p_N, pI_N, pE_N = TrajectoryOptimization.get_num_constraints(solver::Solver)

ui = TrajectoryOptimization.infeasible_controls(solver,X_infeasible,U_infeasible)
results_infeasible = TrajectoryOptimization.ConstrainedVectorResults(solver.model.n,solver.model.m+solver.model.n,p,solver.N,p_N)
copyto!(results_infeasible.U, [U_infeasible;ui])
TrajectoryOptimization.rollout!(results_infeasible,solver)

@test all(ui[1,1:end-1] .== ui[1,1]) # special case for state trajectory of all ones, control 1 should all be same
@test all(ui[2,1:end-1] .== ui[2,1]) # special case for state trajectory of all ones, control 2 should all be same
@test all(TrajectoryOptimization.to_array(results_infeasible.X) == X_infeasible)
# rolled out trajectory should be equivalent to infeasible trajectory after applying augmented controls

### OTHER TESTS ###
# Test undefined integration
@test_throws ArgumentError TrajectoryOptimization.Solver(model,obj_c, integration=:bogus, dt=0.1, opts=opts)
