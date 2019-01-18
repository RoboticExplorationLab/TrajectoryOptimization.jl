using Plots
using Random

Random.seed!(123)

# Solver options
tf = 5.0
N = 101
integration = :rk4
opts = SolverOptions()
opts.verbose = true
opts.cost_tolerance = 1e-8
opts.cost_tolerance_intermediate = 1e-8
opts.constraint_tolerance = 1e-8
opts.constraint_tolerance_intermediate = sqrt(opts.constraint_tolerance)
opts.gradient_norm_tolerance = 1e-12
opts.gradient_norm_tolerance_intermediate = 1e-12
opts.use_gradient_aula = false
opts.active_constraint_tolerance = 0.0
opts.penalty_scaling = 10.0
opts.penalty_initial = 1.0
opts.constraint_decrease_ratio = .25
opts.iterations = 1000
opts.iterations_outerloop = 50
opts.iterations_innerloop = 300
opts.outer_loop_update_type = :feedback

# u_min = -2
# u_max = 2
# model, obj = TrajectoryOptimization.Dynamics.pendulum!
#
# obj_con = TrajectoryOptimization.ConstrainedObjective(obj,u_min=u_min,u_max=u_max)
#
# solver_uncon = Solver(model,obj,integration=integration,N=N,opts=opts)
# solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)

model, = TrajectoryOptimization.Dynamics.dubinscar
n, m = model.n,model.m

x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
dt = 0.01
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con_box = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max,use_xf_equality_constraint=true)

solver_uncon  = Solver(model, obj, integration=integration, N=N, opts=opts)
solver_con = Solver(model, obj_con_box, integration=integration, N=N, opts=opts)

U0 = rand(solver_uncon.model.m, solver_uncon.N-1)
@time results_uncon, stats_uncon = solve(solver_uncon,U0)
@time results_con, stats_con = solve(solver_con,U0)

solver_con
plot(to_array(results_con.U)[:,1:solver_con.N-1]')
plot(to_array(results_con.X)')
plot(to_array(results_con.λ[1:N-1])')

plot(log.(stats_con["max_condition_number"]))
plot(log.(stats_con["c_max"]))
plot(log.(stats_con["cost"]))
plot(log.(stats_con["gradient_norm"]))
