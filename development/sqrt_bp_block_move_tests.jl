using Test
using Plots

opts = TrajectoryOptimization.SolverOptions()
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
opts.iterations = 500
opts.iterations_outerloop = 10
opts.iterations_innerloop = 300
opts.outer_loop_update_type = :default
opts.use_gradient_aula = false

model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -0.1
u_max = 0.1
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=20.0, u_min=u_min, u_max=u_max)#, x_min=x_min, x_max=x_max)

integrator = :rk4
dt = 0.05
solver = TrajectoryOptimization.Solver(model,obj_con,integration=integrator,dt=dt,opts=opts)
get_num_constraints(solver)
get_num_terminal_constraints(solver)
U = zeros(solver.model.m, solver.N-1)
results, stats = TrajectoryOptimization.solve(solver,U)
solver.opts.square_root = true
results_sqrt, stats_sqrt = TrajectoryOptimization.solve(solver,U)

# S_sqrt = [zeros(nn,nn) for k = 1:N]
cond_normal = zeros(N)
cond_sqrt = zeros(N)

for k = 1:N
    # S_sqrt[k] = results2.S[k]'*results2.S[k]
    cond_normal[k] = cond(results.S[k])
    cond_sqrt[k] = cond(results_sqrt.S[k])
end

plot(cond_normal)
plot!(cond_sqrt)
plot(to_array(results.U)',label="")
plot(to_array(results.X)',label="")

plot(log.(stats["gradient_norm"]))

plot(stats["max_condition_number_S"])
plot!(stats_sqrt["max_condition_number_S"])

plot(stats["max_condition_number"])
plot!(stats_sqrt["max_condition_number"])
