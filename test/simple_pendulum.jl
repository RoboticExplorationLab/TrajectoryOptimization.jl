using TrajectoryOptimization
using Base.Test

# Set up models and objective
u_bound = 3.
model,obj = TrajectoryOptimization.Dynamics.pendulum
opts = TrajectoryOptimization.SolverOptions()
opts.c1 = 1e-3
opts.c2 = 2.0
opts.verbose = false
opts.mu_al_update = 100.

obj.Q .= eye(2)*1e-3
obj.R .= eye(1)*1e-2
obj.tf = 5.
model! = TrajectoryOptimization.Model(Dynamics.pendulum_dynamics!,2,1) # inplace dynamics
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective


### UNCONSTRAINED ###
# rk4
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
U = zeros(solver.model.m, solver.N)
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
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache=true
opts.c1=1e-4
opts.c2=2.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-5
opts.iterations_outerloop = 250
opts.iterations = 1000

u_min = -3
u_max = 3
x_min = [-10;-10]
x_max = [10; 10]
obj_uncon = TrajectoryOptimization.Dynamics.pendulum[2]
obj_uncon.R[:] = [1e-2] # control needs to be properly regularized for infeasible start to produce a good warm-start control output

obj_inf = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)
solver = TrajectoryOptimization.Solver(model!, obj_inf, dt=0.1, opts=opts)
X_interp = TrajectoryOptimization.line_trajectory(obj_inf.x0, obj_inf.xf,solver.N)
results_inf = TrajectoryOptimization.solve(solver,X_interp,U)
max_c = TrajectoryOptimization.max_violation(results_inf.result[end])
@test norm(results_inf.X[:,end]-obj.xf) < 1e-3
@test max_c < 1e-2

# test that control output from infeasible start is a good warm start (ie, that infeasible control output is "near" dynamically constrained control output)
idx = find(x->x==2,results_inf.iter_type) # results index where switch from infeasible solve to dynamically constrained solve occurs

# plot(results_inf.result[end].X')
# plot(results_inf.result[idx[1]].U',color="green")
# plot!(results_inf.result[end].U',color="red")

@test norm(results_inf.result[idx[1]].U-results_inf.result[end].U) < 0.5 # confirm that infeasible and final feasible controls are "near"

tmp = TrajectoryOptimization.ConstrainedResults(solver.model.n,solver.model.m,size(results_inf.result[1].C,1),solver.N)
tmp.U[:,:] = results_inf.result[idx[1]].U # store infeasible control output
tmp2 = TrajectoryOptimization.ConstrainedResults(solver.model.n,solver.model.m,size(results_inf.result[1].C,1),solver.N)
tmp2.U[:,:] = results_inf.result[end].U # store

TrajectoryOptimization.rollout!(tmp,solver)
TrajectoryOptimization.rollout!(tmp2,solver)

# plot(tmp.X')
# plot!(tmp2.X')

@test norm(tmp.X[:]-tmp2.X[:]) < 0.5 # test that infeasible state trajectory rollout is "near" dynamically constrained state trajectory rollout

# test linear interpolation for state trajectory
@test norm(X_interp[:,1] - solver.obj.x0) < 1e-8
@test norm(X_interp[:,end] - solver.obj.xf) < 1e-8
@test all(X_interp[1,2:end-1] .<= max(solver.obj.x0[1],solver.obj.xf[1]))
@test all(X_interp[2,2:end-1] .<= max(solver.obj.x0[2],solver.obj.xf[2]))

# test that additional augmented controls can achieve an infeasible state trajectory
U_infeasible = ones(solver.model.m,solver.N)
X_infeasible = ones(solver.model.n,solver.N)
solver.obj.x0 = ones(solver.model.n)
ui = TrajectoryOptimization.infeasible_controls(solver,X_infeasible,U_infeasible)
results_infeasible = TrajectoryOptimization.ConstrainedResults(solver.model.n,solver.model.m+solver.model.n,1,solver.N,1)
results_infeasible.U[:,:] = [U_infeasible;ui]
# solver.opts.infeasible = true  # solver needs to know to use an infeasible rollout
TrajectoryOptimization.rollout!(results_infeasible,solver)

@test all(ui[1,1:end-1] .== ui[1,1]) # special case for state trajectory of all ones, control 1 should all be same
@test all(ui[2,1:end-1] .== ui[2,1]) # special case for state trajectory of all ones, control 2 should all be same
@test all(results_infeasible.X .== X_infeasible)
# rolled out trajectory should be equivalent to infeasible trajectory after applying augmented controls

### OTHER TESTS ###
# Test undefined integration
@test_throws ArgumentError TrajectoryOptimization.Solver(model!,obj_c, integration=:bogus, dt=0.1, opts=opts)
