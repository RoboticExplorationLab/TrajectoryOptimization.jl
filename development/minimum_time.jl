using TrajectoryOptimization
using LinearAlgebra
using Plots
using Logging

## Minimum Time (first-order hold)

# # Pendulum
# u_bound = 5.
# model, obj = TrajectoryOptimization.Dynamics.pendulum!
# opts = TrajectoryOptimization.SolverOptions()
# opts.verbose = false
# obj
# obj.Q = 1e-3*Diagonal(I,2)
# obj.R = 1e-3*Diagonal(I,1)
# obj.tf = 3.
# obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective
# obj_min = update_objective(obj_c,tf=:min,c=0.0, Q = obj.Q, R = obj.R, Qf = obj.Qf*0.0)
# dt = 0.1
# n,m = model.n, model.m
#
# solver = Solver(model,obj_c,integration=:rk3,dt=dt)
# U = ones(m,solver.N)
# results,stats = solve(solver,U)
# plot(to_array(results.X)')
# plot(to_array(results.U)')
#
# solver_min = Solver(model,obj_min,integration=:rk3_foh,N=31)
# U = ones(m,solver_min.N)
# solver_min.opts.verbose = true
# solver_min.opts.use_static = false
# solver_min.opts.max_dt = 0.15
# solver_min.opts.min_dt = 1e-3
# solver_min.opts.constraint_tolerance = 0.001 # 0.005
# solver_min.opts.R_minimum_time = 15.0#13.5 # 12.0
# solver_min.opts.ρ_initial = 0
# solver_min.opts.τ = .25
# solver_min.opts.γ = 2.0
# solver_min.opts.outer_loop_update = :individual
# solver_min.opts.iterations = 100
# solver_min.opts.iterations_outerloop = 25 # 20
# @time results_min,stats_min = solve(solver_min,U)
# total_time(solver_min,results_min)
# plot(to_array(results_min.X)[1:2,:]')
# plot(to_array(results_min.U)[1:2,:]')
# plot(stats_min["cost"])
#
# results_min.U[10][2]^2

u_bound = 20.
model, obj = TrajectoryOptimization.Dynamics.cartpole_analytical
opts = TrajectoryOptimization.SolverOptions()
obj.Q = 1e-3*Diagonal(I,model.n)
obj.R = 1e-3*Diagonal(I,model.m)
obj.tf = 3.
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective
obj_min = TrajectoryOptimization.update_objective(obj_c,tf=:min,c=0.0, Q = obj.Q, R = obj.R, Qf = obj.Qf*0.0)
dt = 0.1
n,m = model.n, model.m

opts.verbose = false
opts.use_static = false
opts.max_dt = 0.1 #0.05
opts.min_dt = 1e-4

# opts.minimum_time_tf_estimate = 0.0
# opts.cost_intermediate_tolerance = 1e-3
# opts.cost_tolerance = 1e-3
opts.constraint_tolerance = 0.001
# opts.gradient_tolerance = 1e-4
# opts.gradient_intermediate_tolerance = 1e-4
opts.R_minimum_time = 1000.0#800.0
opts.ρ_initial = 0
opts.τ = .25
opts.γ = 2.0
opts.outer_loop_update = :individual
opts.iterations = 250
opts.iterations_outerloop = 35 # 20
opts.ρ_forwardpass = 10.0
opts.iterations_linesearch = 20

solver_min = TrajectoryOptimization.Solver(model,obj_min,integration=:rk3_foh,N=31,opts=opts)
U = zeros(m,solver_min.N)

@time results_min,stats_min = TrajectoryOptimization.solve(solver_min,U)
println("final time: $(TrajectoryOptimization.total_time(solver_min,results_min))")
# plot(TrajectoryOptimization.to_array(results_min.X)[1:2,:]')
plot(TrajectoryOptimization.to_array(results_min.U)[1:2,:]')
# plot(stats_min["cost"])

results_min.U[12][2]^2
