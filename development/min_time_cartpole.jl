
u_bound = 30.
model, obj = TrajectoryOptimization.Dynamics.cartpole_udp
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false

obj
obj.Q = 1e-3*Diagonal(I,model.n)
obj.R = 1e-3*Diagonal(I,model.m)
obj.tf = 4.
obj_c = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-u_bound, u_max=u_bound) # constrained objective
obj_min = update_objective(obj_c,tf=:min,c=0.0, Q = obj.Q*0.0, R = obj.R*1.0, Qf = obj.Qf*0.0)
dt = 0.1
n,m = model.n, model.m

solver = Solver(model,obj_c,integration=:rk3,dt=dt)
X_interp = line_trajectory(solver)
U = ones(m,solver.N)
results,stats = solve(solver,U)
plot(to_array(results.X)')
plot(to_array(results.U)')

solver_min = Solver(model,obj_min,integration=:rk3_foh,N=41)
U = ones(m,solver_min.N)
solver_min.opts.verbose = true
solver_min.opts.use_static = false
solver_min.opts.max_dt = 0.2
solver_min.opts.constraint_tolerance = 1e-2
solver_min.opts.R_minimum_time = 100.0
solver_min.opts.ρ_initial = 0
solver_min.opts.τ = .25
solver_min.opts.γ = 10.0
solver_min.opts.iterations_linesearch = 25
solver_min.opts.iterations_outerloop = 30
results_min,stats_min = solve(solver_min,U)
total_time(solver_min,results_min)
plot(to_array(results_min.X)[1:2,:]')
plot(to_array(results_min.U)[1:2,:]')
plot(stats_min["cost"])
