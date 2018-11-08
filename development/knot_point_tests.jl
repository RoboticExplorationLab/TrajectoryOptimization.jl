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
opts.iterations_linesearch = 10
opts.τ = 0.25
opts.γ = 10.0
opts.ρ_initial = 0.0
opts.outer_loop_update = :default
opts.use_static = false
opts.resolve_feasible = false
opts.λ_second_order_update = false
opts.regularization_type = :control
######################

### Set up model, objective, solver ###
# Model, objective (unconstrained)
model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!
model_dubins, obj_uncon_dubins = TrajectoryOptimization.Dynamics.dubinscar!
model_cartpole, obj_uncon_cartpole = TrajectoryOptimization.Dynamics.cartpole_udp
model_quadrotor, obj_uncon_quadrotor = TrajectoryOptimization.Dynamics.quadrotor

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
u_min_cartpole = -20
u_max_cartpole = 40
x_min_cartpole = [-10; -1000; -1000; -1000]
x_max_cartpole = [10; 1000; 1000; 1000]

# quadrotor
u_min = -100.0
u_max = 100.0
# -constraint that quaternion should be unit
function cE(c,x,u)
    c[1] = x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2 - 1.0
    return nothing
end

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)
obj_con_dubins = ConstrainedObjective(obj_uncon_dubins, u_min=u_min_dubins, u_max=u_max_dubins, x_min=x_min_dubins, x_max=x_max_dubins)
obj_con_cartpole = ConstrainedObjective(obj_uncon_cartpole, u_min=u_min_cartpole, u_max=u_max_cartpole, x_min=x_min_cartpole, x_max=x_max_cartpole)
obj_con_quadrotor = TrajectoryOptimization.ConstrainedObjective(obj_uncon_quadrotor, u_min=u_min, u_max=u_max)#, cE=cE)

# System selection
model = model_pendulum
obj = obj_con_pendulum
u_max = u_max_pendulum
u_min = u_min_pendulum

# Solver
intergrator_zoh = :rk3
intergrator_foh = :rk3_foh

dt = 0.2
solver_zoh = Solver(model,obj,integration=intergrator_zoh,dt=dt,opts=opts)
solver_foh = Solver(model,obj,integration=intergrator_foh,dt=dt,opts=opts)

# -Initial state and control trajectories
X_interp = line_trajectory(solver_zoh.obj.x0,solver_zoh.obj.xf,solver_zoh.N)
U = zeros(solver_zoh.model.m,solver_zoh.N)
#######################################

### Solve ###
@time results_zoh, stats_zoh = solve(solver_zoh,U)
@time results_foh, stats_foh = solve(solver_foh,U)

### Results ###
println("Final state (zoh)-> res: $(results_zoh.X[end]), goal: $(solver_zoh.obj.xf)\n Iterations: $(stats_zoh["iterations"])\n Outer loop iterations: $(stats_zoh["major iterations"])\n Max violation: $(stats_zoh["c_max"][end])\n Max μ: $(maximum([to_array(results_zoh.μ)[:]; results_zoh.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_zoh.λ)[:]; results_zoh.λN[:]])))\n")
println("Final state (foh)-> res: $(results_foh.X[end]), goal: $(solver_foh.obj.xf)\n Iterations: $(stats_foh["iterations"])\n Outer loop iterations: $(stats_foh["major iterations"])\n Max violation: $(stats_foh["c_max"][end])\n Max μ: $(maximum([to_array(results_foh.μ)[:]; results_foh.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_foh.λ)[:]; results_foh.λN[:]])))\n")
#
# # Controllers
# controller_zoh = generate_controller(to_array(results_zoh.X),to_array(results_zoh.U),to_array(results_zoh.K),solver_zoh.N,solver_zoh.dt,:zoh,u_min,u_max)
# controller_foh = generate_controller(to_array(results_foh.X),to_array(results_foh.U),to_array(results_foh.K),solver_foh.N,solver_foh.dt,:foh,u_min,u_max)
#
# # Simulate
# dt_sim = 0.001
# tf = obj.tf
# t_solve = 0:dt:tf
# t_sim = 0:dt_sim:tf
# x0 = obj.x0
# f = model.f
# n = model.n
# m = model.m
# integrator_sim = :rk4 # note that ode45 can be used with 'simulate_controller' but the time indicies must be altered when plotting
#
# X_zoh_sim, U_zoh_sim = simulate_controller(f,integrator_sim,controller_zoh,n,m,dt_sim,x0,tf,u_min,u_max)
# X_foh_sim, U_foh_sim = simulate_controller(f,integrator_sim,controller_foh,n,m,dt_sim,x0,tf,u_min,u_max)
#
# # RMS error
# xf_rms_zoh = sqrt(mean((results_zoh.X[end] - X_zoh_sim[:,end]).^2))
# xf_rms_foh = sqrt(mean((results_foh.X[end] - X_foh_sim[:,end]).^2))
#
# length(t_sim)
# # ZOH plotting
# p_zoh = plot(t_solve,to_array(results_zoh.X)',color="black",title="zoh (N = $(solver_zoh.N),dt=$dt,tf=$(solver_zoh.obj.tf)) sim @ $(convert(Int64,1/dt_sim))Hz",labels="")
# for i = 1:n
#     p_zoh = plot!(t_sim,X_zoh_sim[i,:],color="green",labels="",xlabel="xf RMS error: $(round(xf_rms_zoh,digits=5))")
# end
# display(p_zoh)
# savefig("knotpointtest_zoh.png")
#
# p_zoh = plot(t_solve,to_array(results_zoh.U)',color="black",title="zoh (N = $(solver_zoh.N),dt=$dt,tf=$(solver_zoh.obj.tf)) sim @ $(convert(Int64,1/dt_sim))Hz",labels="")
# for i = 1:m
#     p_zoh = plot!(t_sim,U_zoh_sim[i,:],color="green",labels="",xlabel="xf RMS error: $(round(xf_rms_zoh,digits=5))")
# end
# display(p_zoh)
# savefig("knotpointtest_zoh_control.png")
#
#
# # FOH plotting
# p_foh = plot(t_solve,to_array(results_foh.X)',color="black",title="foh (N = $(solver_foh.N),dt=$dt,tf=$(solver_foh.obj.tf)) sim @ $(convert(Int64,1/dt_sim))Hz",labels="")
# for i = 1:n
#     p_foh = plot!(t_sim,X_foh_sim[i,:],color="purple",labels="",xlabel="xf RMS error: $(round(xf_rms_foh,digits=5))")
# end
# display(p_foh)
# savefig("knotpointtest_foh.png")
#
#
# p_foh = plot(t_solve,to_array(results_foh.U)',color="black",title="foh (N = $(solver_foh.N),dt=$dt,tf=$(solver_foh.obj.tf)) sim @ $(convert(Int64,1/dt_sim))Hz",labels="")
# for i = 1:m
#     p_foh = plot!(t_sim,U_foh_sim[i,:],color="purple",labels="",xlabel="xf RMS error: $(round(xf_rms_foh,digits=5))")
# end
# display(p_foh)
# savefig("knotpointtest_foh_control.png")
#
#
#
# # ODE45 simulation
# # using ODE
# # function closed_loop_dynamics(t,x)
# #     xdot = zeros(n)
# #     f(xdot,x,controller_foh(x,t))
# #     xdot
# # end
# #
# # t_sim, X_sim = ode45(closed_loop_dynamics, x0, 0:dt_sim:tf)
# # plot(t_sim,to_array(X_sim)')
