### Solver options ###
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.c1 = 1e-8
opts.c2 = 10.0
opts.cost_intermediate_tolerance = 1e-3
opts.constraint_tolerance = 1e-2
opts.cost_tolerance = 1e-3
opts.iterations_outerloop = 50
opts.iterations = 250
opts.iterations_linesearch = 10
opts.τ = 0.25
opts.γ = 10.0
opts.ρ_initial = 0.0
opts.outer_loop_update = :individual
opts.use_static = false
opts.resolve_feasible = false
opts.λ_second_order_update = false
opts.regularization_type = :control
opts.max_dt = 0.25
opts.min_dt = 0.01
opts.min_time_regularization = 0.0
opts.μ_initial_minimum_time_inequality = 1.0
opts.μ_initial_minimum_time_equality = 1.0
opts.ρ_forwardpass = 5.0
######################

### Set up model, objective, solver ###
# Model, objective (unconstrained)
model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!
model_dubins, obj_uncon_dubins = TrajectoryOptimization.Dynamics.dubinscar!
model_cartpole, obj_uncon_cartpole = TrajectoryOptimization.Dynamics.cartpole_udp
model_quadrotor, obj_uncon_quadrotor = TrajectoryOptimization.Dynamics.quadrotor

## Constraints

# pendulum
u_min_pendulum = -3
u_max_pendulum = 3
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

# quadrotor
u_min_quadrotor = -100.0
u_max_quadrotor = 100.0
# -constraint that quaternion should be unit
function cE(c,x,u)
    c[1] = x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2 - 1.0
    return nothing
end

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)
obj_con_dubins = ConstrainedObjective(obj_uncon_dubins, u_min=u_min_dubins, u_max=u_max_dubins, x_min=x_min_dubins, x_max=x_max_dubins)
obj_con_cartpole = ConstrainedObjective(obj_uncon_cartpole, u_min=u_min_cartpole, u_max=u_max_cartpole, x_min=x_min_cartpole, x_max=x_max_cartpole)
obj_con_quadrotor = TrajectoryOptimization.ConstrainedObjective(obj_uncon_quadrotor, u_min=u_min_quadrotor, u_max=u_max_quadrotor)#, cE=cE)

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
opts.max_dt = dt
obj_mintime = ConstrainedObjective(0.0*obj.Q,Matrix(I,m,m),obj.Qf,0.0,obj.x0,obj.xf,c=1e-5,u_min=obj.u_min,u_max=obj.u_max)
solver_mintime = Solver(model,obj_mintime,integration=:rk3_foh,N=N_mintime,dt=0.0,opts=opts)


U = ones(m,solver.N)
U_mintime = rand(m,N_mintime)

results,stats = solve(solver,U)
results_mintime,stats_mintime = solve(solver_mintime,U_mintime)

val,idx = findmax(to_array(results_mintime.C))
valn,idxn = findmax(results_mintime.CN)
println("Final state (    reg)-> res: $(results.X[end]), goal: $(solver.obj.xf)\n Iterations: $(stats["iterations"])\n Outer loop iterations: $(stats["major iterations"])\n Max violation: $(stats["c_max"][end])\n Max μ: $(maximum([to_array(results.μ)[:]; results.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results.λ)[:]; results.λN[:]])))\n")
println("Final state (mintime)-> res: $(results_mintime.X[end]), goal: $(solver_mintime.obj.xf)\n Iterations: $(stats_mintime["iterations"])\n Outer loop iterations: $(stats_mintime["major iterations"])\n Max violation: $(stats_mintime["c_max"][end])\n Max μ: $(maximum([to_array(results_mintime.μ)[:]; results_mintime.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_mintime.λ)[:]; results_mintime.λN[:]])))\n")

# plot(to_array(results_mintime.U)')
# plot(to_array(results_mintime.X)')
#
# results_mintime.U[end-1]
a = 1

# # initialize regular results
# p,pI,pE = get_num_constraints(solver)
# m̄,mm = get_num_controls(solver)
# results = ConstrainedVectorResults(n,mm,p,N)
# results.μ .*= solver.opts.μ1
# results.ρ[1] = solver.opts.ρ_initial
# copyto!(results.U, U)
# X = results.X # state trajectory
# U = results.U # control trajectory
# X_ = results.X_ # updated state trajectory
# U_ = results.U_ # updated control trajectory
# X[1] = solver.obj.x0
# flag = rollout!(results,solver) # rollout new state trajectory
#
# display(plot(to_array(U)'))
# display(plot(to_array(X)'))
#
# # # initialize mintime results
# # p_mintime,pI_mintime,pE_mintime = get_num_constraints(solver_mintime)
# # m̄_mintime,mm_mintime = get_num_controls(solver_mintime)
# # results_mintime = ConstrainedVectorResults(n,mm_mintime,p_mintime,N)
# # results_mintime.μ .*= solver_mintime.opts.μ1
# # results_mintime.ρ[1] = solver_mintime.opts.ρ_initial
# # copyto!(results_mintime.U, U_mintime)
# # X_mintime = results_mintime.X # state trajectory
# # U_mintime = results_mintime.U # control trajectory
# # X_mintime_ = results_mintime.X_ # updated state trajectory
# # U_mintime_ = results_mintime.U_ # updated control trajectory
# # X_mintime[1] = solver_mintime.obj.x0
# # flag = rollout!(results_mintime,solver_mintime) # rollout new state trajectory
#
# # display(plot(to_array(U_mintime)'))
# # display(plot(to_array(X_mintime)'))
#
# ## normal solve
# # inner loop
# inner_loop(results,solver)
#
# # outerloop
# λ_update!(results,solver,false)
#
# ## Penalty updates
# μ_update!(results,solver)
# #
# # ## mintime solve
# # # inner loop
# # calculate_jacobians!(results_mintime, solver_mintime)
# # Δv = _backwardpass_foh_min_time!(results_mintime, solver_mintime)
# # J = forwardpass!(results_mintime, solver_mintime, Δv)
# # X_mintime .= deepcopy(X_mintime_)
# # U_mintime .= deepcopy(U_mintime_)
# # display(plot(to_array(U_mintime)'))
# # display(plot(to_array(X_mintime)'))
# #
# # # outerloop
# # outer_loop_update(results_mintime,solver_mintime,false)
