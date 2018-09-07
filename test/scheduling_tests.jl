using TrajectoryOptimization
using Plots
using Base.Test

dt = 0.01
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
# opts.c1 = 1e-4
# opts.c2 = 2.0
# opts.mu_al_update = 10.0
opts.eps_intermediate = 1e-1
opts.eps_constraint = 1e-5
opts.eps = 1e-5
opts.iterations_outerloop = 50
opts.iterations = 500
# opts.iterations_linesearch = 50
opts.τ = 0.1#1e-1
opts.outer_loop_update = :uniform
# System
model, obj_uncon = TrajectoryOptimization.Dynamics.pendulum!

u_min = -2
u_max = 2
x_min = [-20;-20]
x_max = [20; 20]
obj_con = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

solver = Solver(model,obj_con,integration=:rk3_foh,dt=dt,opts=opts)

# test linear interpolation for state trajectory
X_interp = ones(solver.model.n,solver.N)
X_interp = line_trajectory(solver.obj.x0,solver.obj.xf,solver.N).*(1 + 0.05*rand(solver.model.n,solver.N))
U = ones(solver.model.m,solver.N)
plot(X_interp')

println("---New test run---")
@time results,stats = solve(solver,X_interp,U)

plot(results.X',title="Pendulum (with constrained control and states (inplace dynamics))",ylabel="x(t)")
plot(results.U',title="Pendulum (with constrained control and states (inplace dynamics))",ylabel="u(t)")
println("Final state: $(results.X[:,end])\n Iterations: $(stats["iterations"])\n Max violation: $(max_violation(results.result[results.termination_index]))")
@test norm(results.X[:,end] - solver.obj.xf) < 1e-5

# plot(log.(results.cost[1:results.termination_index]))
