# Pendulum
model,obj = TrajectoryOptimization.Dynamics.pendulum
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
solver_min.opts.R_minimum_time = 15.0 #15.0 #13.5 # 12.0
solver_min.opts.constraint_decrease_ratio = .25
solver_min.opts.penalty_scaling = 2.0
solver_min.opts.outer_loop_update_type = :default
solver_min.opts.iterations = 1000
solver_min.opts.iterations_outerloop = 50 # 20

U = ones(m,solver_min.N-1)
results_uncon,stats_uncon = TrajectoryOptimization.solve(solver_uncon,U)
results_min,stats_min = TrajectoryOptimization.solve(solver_min,U)
# plot(TrajectoryOptimization.to_array(results_min.X)[1:2,:]')
plot(TrajectoryOptimization.to_array(results_min.U)[1:2,:]')

T_uncon = TrajectoryOptimization.total_time(solver_uncon,results_uncon)
T_min_time = TrajectoryOptimization.total_time(solver_min,results_min)

@test T_min_time < 0.5*T_uncon
@test T_min_time < 1.5
@test norm(results_min.X[end][1:n] - obj.xf) < 1e-3
@test TrajectoryOptimization.max_violation(results_min) < solver_min.opts.constraint_tolerance

#######################
## Box Parallel Park ##
######################

# Solver Options
dt = 0.01
integration = :rk4

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-4
opts.cost_tolerance_intermediate = 1e-3

# Set up model, objective, and solver
N = 51
model, = TrajectoryOptimization.Dynamics.dubinscar
n, m = model.n,model.m

x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

u_max = 2.0
u_min = -2.0
obj_con_box = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max,u_max=u_max,u_min=u_min)

solver_uncon  = Solver(model, obj, integration=integration, N=N, opts=opts)
solver_con_box = Solver(model, obj_con_box, integration=integration, N=N, opts=opts)

U0 = ones(solver_con_box.model.m,solver_con_box.N-1)
# X0 = line_trajectory(solver_con_box)

# results_uncon, stats_uncon = TrajectoryOptimization.solve(solver_uncon,U0)
results_con_box, stats_con_box = TrajectoryOptimization.solve(solver_con_box,U0)

obj_mintime = update_objective(obj_con_box,tf=:min, u_min=u_min, u_max=u_max)

opts.max_dt = 0.2
opts.min_dt = 1e-3
opts.minimum_time_dt_estimate = tf/(N-1)
opts.constraint_tolerance = 0.001 # 0.005
opts.R_minimum_time = .05 #15.0 #13.5 # 12.0
opts.constraint_decrease_ratio = .25
opts.penalty_scaling = 10.0
opts.outer_loop_update_type = :individual
opts.iterations = 1000
opts.iterations_outerloop = 30 # 20

solver_mintime = Solver(model, obj_mintime, integration=integration, N=N, opts=opts)
solver_mintime.state.penalty_only = false
results_mintime, stats_mintime = solve(solver_mintime,to_array(results_con_box.U))
solver_mintime
T = TrajectoryOptimization.total_time(solver_con_box,results_con_box)
T_min = TrajectoryOptimization.total_time(solver_mintime,results_mintime)
# plot(to_array(results_mintime.U)[1:2,1:solver_mintime.N-1]',labels="")
# plot(to_array(results_mintime.X)[1:3,:]',labels="")
# plot(to_array(results_mintime.X)[1,:],to_array(results_mintime.X)[2,:],width=2,color=:blue,label="Minimum Time")

@test max_violation(results_mintime) < solver_mintime.opts.constraint_tolerance
@test T_min < 0.75*T
@test norm(results_mintime.X[end][1:n] - obj_con_box.xf) < 1e-3
@test T_min < 1.75
