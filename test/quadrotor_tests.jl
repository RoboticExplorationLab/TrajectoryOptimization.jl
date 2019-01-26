Random.seed!(123)

# Solver options
N = 201
integration = :rk4
opts = SolverOptions()
opts.verbose = false
opts.square_root = true
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5

# Set up model, objective, solver
model, obj = TrajectoryOptimization.Dynamics.quadrotor
n = model.n
m = model.m

# Unconstrained solve
solver_uncon = Solver(model,obj,integration=integration,N=N,opts=opts)
U0 = ones(solver_uncon.model.m, solver_uncon.N-1)
results_uncon, stats_uncon = solve(solver_uncon,U0)
@test norm(results_uncon.X[end]-obj.xf) < 5e-3

# Unit quaternion
model, obj_uq = TrajectoryOptimization.Dynamics.quadrotor_unit_quaternion

solver_uq = Solver(model,obj_uq,integration=integration,N=N,opts=opts)
results_uq, stats_uq = solve(solver_uq,U0)

@test norm(results_uq.X[end]-obj_uq.xf) < 1e-5
@test norm(max_violation(results_uq)) < 1e-5

# 3 sphere obstacles + unit quaternion
model, obj_obs = TrajectoryOptimization.Dynamics.quadrotor_3obs

solver_obs = Solver(model,obj_obs,integration=integration,N=N,opts=opts)
solver_obs.opts.square_root = true
solver_obs.opts.outer_loop_update_type = :feedback
results_obs, stats_obs = solve(solver_obs,U0)

@test norm(results_obs.X[end]-obj_obs.xf) < 1e-5
@test norm(max_violation(results_obs)) < 1e-5

# plot(to_array(results_obs.X)[1:3,:]')
# plot(to_array(results_obs.U[1:solver_obs.N-1])')
