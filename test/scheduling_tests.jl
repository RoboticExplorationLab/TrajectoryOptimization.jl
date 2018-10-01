# using PyPlot
using TrajectoryOptimization
### Solver options ###
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
# opts.c1 = 1e-4
opts.c2 = 5.0
opts.cost_intermediate_tolerance = 1e-5
opts.constraint_tolerance = 1e-5
opts.cost_tolerance = 1e-5
opts.iterations_outerloop = 50
opts.iterations = 500
# opts.iterations_linesearch = 50
opts.τ = 0.25
opts.ρ_initial = 0.0
opts.outer_loop_update = :sequential
######################

### Set up model, objective, solver ###
# Model, objective (unconstrained)
model, obj_uncon = TrajectoryOptimization.Dynamics.pendulum!

# -Constraints
u_min = -2
u_max = 2
x_min = [-20;-20]
x_max = [20; 20]

# -Constrained objective
obj_con = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

# Solver
intergrator = :rk3_foh
opts.use_static = false
opts.resolve_feasible = true
opts.λ_second_order_update = false
solver = Solver(model,obj_con,integration=intergrator,dt=dt,opts=opts)

opts2 = copy(opts)
opts2.λ_second_order_update = true
solver2 = Solver(model,obj_con,integration=intergrator,dt=dt,opts=opts2)

# -Initial state and control trajectories
X_interp = line_trajectory(solver.obj.x0,solver.obj.xf,solver.N)
U = ones(solver.model.m,solver.N)
#######################################

### Solve ###
@time results,stats = solve(solver,U)
@time results2,stats2 = solve(solver2,U)
@time results3,stats3 = solve(solver,X_interp,U)
@time results4,stats4 = solve(solver2,X_interp,U)

### Results ###
println("Final state (1st): $(results.X[end])\n Iterations: $(stats["iterations"])\n Max violation: $(max_violation(results.result[results.termination_index]))")
println("Final state (2nd): $(results2.X[end])\n Iterations: $(stats2["iterations"])\n Max violation: $(max_violation(results2.result[results2.termination_index]))")
println("Final state (1st inf): $(results3.X[end])\n Iterations: $(stats3["iterations"])\n Max violation: $(max_violation(results3.result[results3.termination_index]))")
println("Final state (2nd inf): $(results4.X[end])\n Iterations: $(stats4["iterations"])\n Max violation: $(max_violation(results4.result[results4.termination_index]))")
#
# results3.result[results3.termination_index]
#
# x = 1
#
# results_tmp = no_infeasible_control_results(results3.result[results3.termination_index],solver)
#
# update_constraints!(results_tmp,solver)
# calculate_jacobians!(results_tmp,solver)
# backwardpass_foh!(results_tmp,solver)
# rollout!(results_tmp,solver)
#
#
# results
#
# ###############
# using PyPlot
# PyPlot.figure()
# iters = range(0,step=solver.dt,length=solver.N)
# # iters2 = range(0,step=solver2.dt,length=solver2.N)
# PyPlot.plot(iters, to_array(results3.X)',label="1st order")
# PyPlot.plot(iters, to_array(results_tmp.X)',label="1st order (trans.)")
#
# # PyPlot.plot(iters2, to_array(results2.X)',label="2nd order")
# PyPlot.xlabel("time step")
# PyPlot.ylabel("state")
# PyPlot.legend()
# PyPlot.title("Pendulum (w/ control and state constraints)")
# PyPlot.show()
# #
# #
# # PyPlot.figure()
# # iters = range(0,step=1,length=stats["iterations"]+1)
# # iters2 = range(0,step=1,length=stats2["iterations"]+1)
# # # PyPlot.plot(iters, log.(results.cost[1:results.termination_index] .+ -1*minimum(results.cost[1:results.termination_index])),label="1st order")
# # # PyPlot.plot(iters2, log.(results2.cost[1:results2.termination_index] .+ -1*minimum(results2.cost[1:results2.termination_index])),label="2nd order")
# # # PyPlot.plot(iters, results.cost[1:results.termination_index],label="1st order")
# # # PyPlot.plot(iters2, results2.cost[1:results2.termination_index],label="2nd order")
# # PyPlot.plot(iters, log.(results.cost[1:results.termination_index]),label="1st order")
# # PyPlot.plot(iters2, log.(results2.cost[1:results2.termination_index]),label="2nd order")
# # PyPlot.xlabel("Iteration")
# # PyPlot.ylabel("log(cost)")
# # PyPlot.legend()
# # PyPlot.title("Pendulum (infeasible start w/ control and state constraints)")
# # PyPlot.show()
# #
# # results.cost[1:results.termination_index]
# # length(iters)
# #
# # results2.cost
# # length(iters2)
# #
# # stats2["λ_second_order"]
