### Solver options ###
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.z_min = 1e-8
opts.z_max = 10.0
opts.cost_intermediate_tolerance = 1e-3
opts.constraint_tolerance = 1e-3
opts.cost_tolerance = 1e-3
opts.iterations_outerloop = 50
opts.iterations = 250
opts.iterations_linesearch = 25
opts.τ = 0.75
opts.γ = 2.0
opts.ρ_initial = 0.0
opts.outer_loop_update = :default
opts.use_static = false
opts.resolve_feasible = false
opts.λ_second_order_update = false
opts.regularization_type = :control
opts.max_dt = 0.25
opts.min_dt = 0.01
opts.μ_initial_minimum_time_inequality = 100.0
opts.μ_initial_minimum_time_equality = 100.0
opts.ρ_forwardpass = 5.0
opts.gradient_tolerance=1e-4
opts.gradient_intermediate_tolerance=1e-4
opts.μ_max = 10000.0
opts.μ_initial = 1.0
opts.λ_max = 1e8
opts.λ_min = -1e8
opts.ρ_max = 1e8
opts.R_minimum_time = 10.0
opts.R_infeasible = 1e-3
opts.resolve_feasible = false
opts.live_plotting = false

######################

### Set up model, objective, solver ###
# Model, objective (unconstrained)
model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!
model_dubins, obj_uncon_dubins = TrajectoryOptimization.Dynamics.dubinscar!
model_cartpole, obj_uncon_cartpole = TrajectoryOptimization.Dynamics.cartpole_udp

## Constraints

# pendulum
u_min_pendulum = -2
u_max_pendulum = 2
x_min_pendulum = [-20;-20]
x_max_pendulum = [20; 20]

# dubins car
u_min_dubins = [-1; -1]
u_max_dubins = [1; 1]
x_min_dubins = [0; -100; -100]
x_max_dubins = [1.0; 100; 100]

# cartpole
u_min_cartpole = -30
u_max_cartpole = 30
x_min_cartpole = [-10; -1000; -1000; -1000]
x_max_cartpole = [10; 1000; 1000; 1000]

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum)#, x_min=x_min_pendulum, x_max=x_max_pendulum)
obj_con_dubins = ConstrainedObjective(obj_uncon_dubins, u_min=u_min_dubins, u_max=u_max_dubins, x_min=x_min_dubins, x_max=x_max_dubins)
obj_con_cartpole = ConstrainedObjective(obj_uncon_cartpole, u_min=u_min_cartpole, u_max=u_max_cartpole, x_min=x_min_cartpole, x_max=x_max_cartpole)

# System selection
model = model_pendulum
n = model.n
m = model.m
obj = obj_con_pendulum
u_max = u_max_pendulum
u_min = u_min_pendulum

dt = 0.2
solver = Solver(model,obj,integration=:rk3_foh,dt=dt,opts=opts)

N_mintime = solver.N
opts.minimum_time_tf_estimate = solver.obj.tf
obj_mintime = ConstrainedObjective(0.0*obj.Q,(1e-5)*Matrix(I,m,m),obj.Qf,0.0,obj.x0,obj.xf,u_min=obj.u_min,u_max=obj.u_max)
solver_mintime = Solver(model,obj_mintime,integration=:rk3_foh,N=N_mintime,opts=opts)
opts.max_dt
opts.min_dt
U = zeros(m,solver.N)
U_mintime = zeros(m,N_mintime)
X_interp = line_trajectory(solver_mintime)

# U_dt = [U; ones(1,size(U,2))*sqrt(get_initial_dt(solver_mintime))]
# ui = infeasible_controls(solver,X_interp,U_dt)
# maximum(ui)
# minimum(ui)
# U_ = [Um; Ui]
U_mintime[1,:] = [val for (i,val) in enumerate(range(u_min,length=solver.N,stop=u_max))]
# U_mintime[1,:] = [u_max*cos(val) for (i,val) in enumerate(range(0,length=solver.N,stop=2*pi))]

# results,stats = solve(solver,U)
results_mintime, stats_mintime = solve(solver_mintime,U_mintime)#to_array(results_mintime.U)[1:m,:])#U_mintime)

# println("Final state (    reg)-> res: $(results.X[end]), goal: $(solver.obj.xf)\n Iterations: $(stats["iterations"])\n Outer loop iterations: $(stats["major iterations"])\n Max violation: $(stats["c_max"][end])\n Max μ: $(maximum([to_array(results.μ)[:]; results.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results.λ)[:]; results.λN[:]])))\n")
println("Final state (mintime)->: $(results_mintime.X[end]), goal: $(solver_mintime.obj.xf)\n Iterations: $(stats_mintime["iterations"])\n Outer loop iterations: $(stats_mintime["major iterations"])\n Max violation: $(stats_mintime["c_max"][end])\n Max μ: $(maximum([to_array(results_mintime.μ)[:]; results_mintime.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_mintime.λ)[:]; results_mintime.λN[:]])))\n")

# p1 = plot(to_array(results_mintime.X)',ylabel="state",label="")
# p2 = plot(to_array(results_mintime.U)[:,1:solver_mintime.N-1]',ylabel="control",xlabel="time step (k)",label="")
# plot(p1,p2,layout=(2,1))

# p1 = plot(to_array(results.X)',ylabel="state",label="")
# p2 = plot(to_array(results.U)[:,1:solver.N-1]',ylabel="control",xlabel="time step (k)",label="")
# plot(p1,p2,layout=(2,1))
