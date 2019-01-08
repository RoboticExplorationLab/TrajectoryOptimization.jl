# Pendulum
model,obj = TrajectoryOptimization.Dynamics.pendulum!
n,m = model.n, model.m

u_bound = 5.
Q_min_time = Array(1e-3*Diagonal(I,n))
R_min_time = Array(1e-3*Diagonal(I,m))
Qf_min_time = Array(Diagonal(I,n)*0.0)
tf = obj.tf
x0 = obj.x0
xf = obj.xf


obj_min_time = TrajectoryOptimization.ConstrainedObjective(LQRObjective(Q_min_time, R_min_time, Qf_min_time, tf, x0, xf), tf=:min, u_min=-u_bound, u_max=u_bound)

solver_uncon = TrajectoryOptimization.Solver(model,obj,integration=:rk4,N=31)
solver_min = TrajectoryOptimization.Solver(model,obj_min_time,integration=:rk4,N=31)

solver_min.opts.verbose = false
solver_min.opts.max_dt = 0.15
solver_min.opts.min_dt = 1e-3
solver_min.opts.constraint_tolerance = 0.001 # 0.005
solver_min.opts.R_minimum_time = 10.0 #15.0 #13.5 # 12.0
solver_min.opts.constraint_decrease_ratio = .25
solver_min.opts.penalty_scaling = 2.0
solver_min.opts.outer_loop_update_type = :individual
solver_min.opts.iterations = 1000
solver_min.opts.iterations_outerloop = 25 # 20

U = ones(m,solver_min.N-1)
results_uncon,stats_uncon = TrajectoryOptimization.solve(solver_uncon,U)
results_min,stats_min = TrajectoryOptimization.solve(solver_min,U)

T_uncon = TrajectoryOptimization.total_time(solver_uncon,results_uncon)
T_min_time = TrajectoryOptimization.total_time(solver_min,results_min)

@test T_min_time < 0.5*T_uncon
@test T_min_time < 1.5
@test norm(results_min.X[end][1:n] - obj.xf) < 1e-3
@test TrajectoryOptimization.max_violation(results_min) < solver_min.opts.constraint_tolerance
# plot(TrajectoryOptimization.to_array(results_min.X)[1:2,:]')
# plot(TrajectoryOptimization.to_array(results_min.U)[1:2,:]')

# ## Cartpole
# model,obj = TrajectoryOptimization.Dynamics.cartpole_analytical
# n, m = model.n, model.m
# Q_min_time = Array(1e-3*Diagonal(I,n))
# R_min_time = Array(1e-3*Diagonal(I,m))
# Qf_min_time = Array(Diagonal(I,n)*0.0)
# u_bound = 15.
#
# obj_min_time = TrajectoryOptimization.ConstrainedObjective(LQRObjective(Q_min_time, R_min_time, Qf_min_time, obj.tf, obj.x0, obj.xf), tf=:min, u_min=-u_bound, u_max=u_bound) # constrained objective
#
# opts = TrajectoryOptimization.SolverOptions()
# opts.verbose = false
# opts.max_dt = 0.1
# opts.min_dt = 1e-3
# opts.constraint_tolerance = 0.001
# opts.R_minimum_time = 10.0 #1000.0
# opts.penalty_initial = 1.0
# opts.bp_reg_initial = 0
# opts.constraint_decrease_ratio = .25
# opts.penalty_scaling = 2.0
# opts.outer_loop_update_type = :individual
# opts.iterations = 1000
# opts.iterations_outerloop = 100 # 20
# opts.bp_reg_fp = 0.0
# opts.iterations_linesearch = 20
#
#
# solver = TrajectoryOptimization.Solver(model,obj,integration=:rk4,N=31,opts=opts)
# U = ones(model.m,solver.N-1)
# results,stats = TrajectoryOptimization.solve(solver,U)
# solver_min = TrajectoryOptimization.Solver(model,obj_min_time,integration=:rk4,N=31,opts=opts)
# results_min,stats_min = TrajectoryOptimization.solve(solver_min,U)#to_array(results.U))
#
#
# T_uncon = TrajectoryOptimization.total_time(solver,results)
#
# T_min_time = TrajectoryOptimization.total_time(solver_min,results_min)
# @test T_min_time < 0.5*T_uncon
# @test T_min_time < 1.5
# @test norm(results_min.X[end][1:n] - obj.xf) < 1e-3
# @test max_violation(results_min) < solver_min.opts.constraint_tolerance
# plot(TrajectoryOptimization.to_array(results_min.X)[1:2,:]')
# plot(TrajectoryOptimization.to_array(results_min.U)[1:2,1:solver_min.N-1]')
#
# results_min.X[end]
#
