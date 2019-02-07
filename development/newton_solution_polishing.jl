using Test
using BenchmarkTools
using Plots
using SparseArrays

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-6
opts.constraint_tolerance = 1e-2
opts.square_root = true
opts.outer_loop_update_type = :default
opts.live_plotting = false
###
model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -0.2
u_max = 0.2
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=5.0,use_xf_equality_constraint=true, u_min=u_min, u_max=u_max)#, x_min=x_min, x_max=x_max)
###

###
model, obj = TrajectoryOptimization.Dynamics.pendulum
obj_con = Dynamics.pendulum_constrained[2]
obj_con.u_min[1] = -1
###

###
model, obj_con = TrajectoryOptimization.Dynamics.dubinscar_parallelpark
###

solver = TrajectoryOptimization.Solver(model,obj_con,integration=:rk4,N=100,opts=opts)
U = rand(solver.model.m, solver.N)

results, stats = TrajectoryOptimization.solve(solver,U)
@assert max_violation(results) < opts.constraint_tolerance
# plot(results.X,title="Block push to origin",xlabel="time step",ylabel="state",label=["pos.";"vel."])
# plot(results.U,title="Block push to origin",xlabel="time step",ylabel="control")
stats["cost"][end]
stats["c_max"][end]

newton_results = NewtonResults(solver)
results_newton, J_newton, c_max_newton = newton_solve(results,newton_results,solver,1.0)

@benchmark newton_solve($results,$newton_results,$solver,$1.0)

# plot(results_newton.U,title="Pendulum",xlabel="time step",ylabel="control",label="Newton",legend=:bottomright)
# plot!(results.U,label="AuLa")

x_min = obj_con.x_min
x_max = obj_con.x_max
plt = plot(title="Parallel Park")#,aspect_ratio=:equal)
plot!(x_min[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(x_max[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_min[2]*ones(1000),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_max[2]*ones(1000),color=:red,width=2,label="")
plot_trajectory!(to_array(results_newton.X),width=2,color=:blue,label="Newton",legend=:bottomright)
plot_trajectory!(to_array(results.X),width=2,color=:green,label="AuLa",legend=:bottomright)


# Cost
cost(solver,results)
J_newton
# Final max constraint tolerance
max_violation(results)
max_violation(results_newton)
