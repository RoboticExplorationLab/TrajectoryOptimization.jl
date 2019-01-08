using TrajectoryOptimization
using LinearAlgebra
using Plots
using Logging

#####################################
## Minimum Time - First-order hold ##
#####################################
# NOTES
#   - Nov. 28, 2018 -T
#   - Found that increasing regularization by +10 after failed fp works well
#   - It's key to not perform an outerloop update when fp fails but to increase reg. as described above, this means convergence for dJ must be > 0.0
#   - the solves are very sensitive to initial conditions, namely max_dt since the initial dt = max_dt/2 and the initial controls
#####################################

# Pendulum
model,obj = TrajectoryOptimization.Dynamics.pendulum!
n,m = model.n, model.m

u_bound = 5.
Q = Array(1e-3*Diagonal(I,n))
R = Array(1e-3*Diagonal(I,m))
Qf = Array(Diagonal(I,n)*0.0)
tf = obj.tf
x0 = obj.x0
xf = obj.xf

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_c = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=-u_bound, u_max=u_bound) # constrained objective
obj_min = TrajectoryOptimization.update_objective(obj_c, tf=:min)

solver_min = TrajectoryOptimization.Solver(model,obj_min,integration=:rk4,N=31)

solver_min.opts.verbose = false
solver_min.opts.use_static = false
solver_min.opts.max_dt = 0.15
solver_min.opts.min_dt = 1e-3
solver_min.opts.constraint_tolerance = 0.001 # 0.005
solver_min.opts.R_minimum_time = 10.0#15.0 #13.5 # 12.0
solver_min.opts.bp_reg_initial = 0
solver_min.opts.constraint_decrease_ratio = .25
solver_min.opts.penalty_scaling = 2.0
solver_min.opts.outer_loop_update_type = :individual
solver_min.opts.iterations = 1000
solver_min.opts.iterations_outerloop = 25 # 20

U = ones(m,solver_min.N-1)
@time results_min,stats_min = TrajectoryOptimization.solve(solver_min,U)

println("Pendulum tf (minimum time): $(TrajectoryOptimization.total_time(solver_min,results_min))")

plot(TrajectoryOptimization.to_array(results_min.X)[1:2,:]')
plot(TrajectoryOptimization.to_array(results_min.U)[1:2,:]')
plot(stats_min["cost"])

# # Cartpole
# model,obj = TrajectoryOptimization.Dynamics.cartpole_analytical
# n, m = model.n, model.m
# Q = Array(0.0*Diagonal(I,n))
# R = Array(1e-3*Diagonal(I,m))
# Qf = Array(Diagonal(I,n)*0.0)
# u_bound = 20.
#
# obj_uncon = LQRObjective(Q, R, Qf, obj.tf, obj.x0, obj.xf)
# obj_c = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=-u_bound, u_max=u_bound) # constrained objective
# obj_min = TrajectoryOptimization.update_objective(obj_c, tf=:min)
#
# opts = TrajectoryOptimization.SolverOptions()
# opts.verbose = false
# opts.max_dt = 0.1
# opts.min_dt = 1e-3
# opts.constraint_tolerance = 0.001
# opts.R_minimum_time = 10.0 #1000.0
# opts.penalty_initial = 1.0
# opts.bp_reg_initial = 0.0
# opts.constraint_decrease_ratio = .25
# opts.penalty_scaling = 2.0
# opts.outer_loop_update_type = :default
# opts.iterations = 500
# opts.iterations_outerloop = 50 # 20
# opts.bp_reg_fp = 10.0
# opts.iterations_linesearch = 20
#
# solver = TrajectoryOptimization.Solver(model,obj,integration=:rk4,N=31,opts=opts)
# U = zeros(model.m,solver.N-1)
# results,stats = TrajectoryOptimization.solve(solver,U)
# solver_min = TrajectoryOptimization.Solver(model,obj_min,integration=:rk4,N=31,opts=opts)
# @time results_min,stats_min = TrajectoryOptimization.solve(solver_min,U)
#
# println("Cartpole tf (minimum time) : $(TrajectoryOptimization.total_time(solver_min,results_min))")
# plot(TrajectoryOptimization.to_array(results_min.X)[1:2,:]')
# plot(TrajectoryOptimization.to_array(results_min.U)[1:2,1:solver_min.N-1]')
# plot(stats_min["cost"])
# max_violation(results_min)
#
#
