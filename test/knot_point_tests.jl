using TrajectoryOptimization
using Random
using Interpolations
using Plots

### Solver options ###
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = false
opts.c1 = 1e-8
opts.c2 = 10.0
opts.cost_intermediate_tolerance = 1e-4
opts.constraint_tolerance = 1e-4
opts.cost_tolerance = 1e-4
opts.iterations_outerloop = 50
opts.iterations = 250
opts.iterations_linesearch = 25
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

# Solver
intergrator_zoh = :rk3
intergrator_foh = :rk3_foh

dt = 0.05
solver_zoh = Solver(model,obj,integration=intergrator_zoh,dt=dt,opts=opts)
solver_foh = Solver(model,obj,integration=intergrator_foh,dt=dt,opts=opts)

# -Initial state and control trajectories
X_interp = line_trajectory(solver_zoh.obj.x0,solver_zoh.obj.xf,solver_zoh.N)
U = rand(solver_zoh.model.m,solver_zoh.N)

#######################################

### Solve ###
@time results_zoh, stats_zoh = solve(solver_zoh,U)
@time results_foh, stats_foh = solve(solver_foh,U)

### Results ###
println("Final state (zoh)-> res: $(results_zoh.X[end]), goal: $(solver_zoh.obj.xf)\n Iterations: $(stats_zoh["iterations"])\n Outer loop iterations: $(stats_zoh["major iterations"])\n Max violation: $(stats_zoh["c_max"][end])\n Max μ: $(maximum([to_array(results_zoh.MU)[:]; results_zoh.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_zoh.LAMBDA)[:]; results_zoh.λN[:]])))\n")
println("Final state (foh)-> res: $(results_foh.X[end]), goal: $(solver_foh.obj.xf)\n Iterations: $(stats_foh["iterations"])\n Outer loop iterations: $(stats_foh["major iterations"])\n Max violation: $(stats_foh["c_max"][end])\n Max μ: $(maximum([to_array(results_foh.MU)[:]; results_foh.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_foh.LAMBDA)[:]; results_foh.λN[:]])))\n")

## prep matrices
K_zoh = [results_zoh.K[k] for k = 1:solver_zoh.N-1]
X_zoh = [results_zoh.X[k] for k = 1:solver_zoh.N]
U_zoh = [results_zoh.U[k] for k = 1:solver_zoh.N-1]

K_foh = [results_foh.K[k] for k = 1:solver_foh.N]
X_foh = [results_foh.X[k] for k = 1:solver_foh.N]
U_foh = [results_foh.U[k] for k = 1:solver_foh.N]

## simulate
dt_sim = 0.001
X_zoh_sim, U_zoh_sim = simulate_lqr_tracking(solver_zoh.model.f,:rk3,dt_sim,X_zoh,U_zoh,K_zoh,solver_zoh.obj.x0,solver_zoh.obj.tf)
X_foh_sim, U_foh_sim = simulate_lqr_tracking(solver_foh.model.f,:rk3,dt_sim,X_foh,U_foh,K_foh,solver_foh.obj.x0,solver_foh.obj.tf)

# rms error on final state
xf_rms_zoh = sqrt(mean((X_zoh[end] - X_zoh_sim[:,end]).^2))
xf_rms_foh = sqrt(mean((X_foh[end] - X_foh_sim[:,end]).^2))

# zoh case
p_zoh = plot(to_array(results_zoh.X)',color="black",title="zoh (N = $(solver_foh.N),dt=$dt,tf=$(solver_zoh.obj.tf)) sim @ $(convert(Int64,1/dt_sim))Hz",labels="")
for i = 1:solver_zoh.model.n
    p_zoh = plot!(range(0,length=convert(Int64,floor(solver_zoh.obj.tf/dt_sim))+1,stop=solver_zoh.N),X_zoh_sim[i,:],color="green",labels="",xlabel="xf RMS error: $(round(xf_rms_zoh,digits=5))")
end
display(p_zoh)
# savefig("knotpointtest_zoh.png")


# foh case
p_foh = plot(to_array(results_foh.X)',color="black",title="foh (N = $(solver_foh.N),dt=$dt,tf=$(solver_foh.obj.tf))) sim @ $(convert(Int64,1/dt_sim))Hz",label="")
for i = 1:solver_foh.model.n
    p_foh = plot!(range(0,length=convert(Int64,floor(solver_foh.obj.tf/dt_sim))+1,stop=solver_foh.N),X_foh_sim[i,:],color="purple",label="",xlabel="xf RMS error: $(round(xf_rms_foh,digits=5))")
end
display(p_foh)
# savefig("knotpointtest_foh.png")
