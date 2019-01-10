using Test
using Plots

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = true

model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -0.1
u_max = 0.1
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=20.0, u_min=u_min, u_max=u_max)#, x_min=x_min, x_max=x_max)

integrator = :rk3
dt = 0.05
solver = TrajectoryOptimization.Solver(model,obj_con,integration=integrator,dt=dt,opts=opts)
get_num_constraints(solver)
get_num_terminal_constraints(solver)
solver.state.second_order_dual_update = false
solver.opts.use_second_order_dual_update = false
solver.opts.use_gradient_aula = true
U = zeros(solver.model.m, solver.N)
results, stats = TrajectoryOptimization.solve(solver,U)

plot(to_array(results.U)',label="")
plot(to_array(results.X)',label="")

plot!(stats["gradient_norm"])

max_violation(results)
maximum(abs.(to_array(results.U)))
