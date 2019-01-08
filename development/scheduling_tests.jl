using TrajectoryOptimization
using Random
using BenchmarkTools
Random.seed!(7)
### Solver options ###
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache = false
opts.c1 = 1e-8
opts.c2 = 10.0
opts.cost_tolerance_intermediate = 1e-4
opts.constraint_tolerance = 1e-4
opts.cost_tolerance = 1e-4
opts.iterations_outerloop = 50
opts.iterations = 250
opts.iterations_linesearch = 25
opts.constraint_decrease_ratio = 0.25
opts.penalty_scaling = 10.0
opts.bp_reg_initial = 0.0
opts.outer_loop_update_type = :individual
opts.use_static = false
opts.resolve_feasible = false
opts.λ_second_order_update = false
opts.bp_reg_type = :control
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
u_min_cartpole = -20
u_max_cartpole = 40
x_min_cartpole = [-10; -1000; -1000; -1000]
x_max_cartpole = [10; 1000; 1000; 1000]

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)
obj_con_dubins = ConstrainedObjective(obj_uncon_dubins, u_min=u_min_dubins, u_max=u_max_dubins, x_min=x_min_dubins, x_max=x_max_dubins)
obj_con_cartpole = ConstrainedObjective(obj_uncon_cartpole, u_min=u_min_cartpole, u_max=u_max_cartpole, x_min=x_min_cartpole, x_max=x_max_cartpole)

# Solver
intergrator = :rk3_foh
solver_pendulum = Solver(model_pendulum,obj_con_pendulum,integration=intergrator,dt=dt,opts=opts)
solver_dubins = Solver(model_dubins,obj_con_dubins,integration=intergrator,dt=dt,opts=opts)
solver_cartpole = Solver(model_cartpole,obj_con_cartpole,integration=intergrator,dt=dt,opts=opts)

# -Initial state and control trajectories
X_interp_pendulum = line_trajectory(solver_pendulum.obj.x0,solver_pendulum.obj.xf,solver_pendulum.N)
X_interp_dubins = line_trajectory(solver_dubins.obj.x0,solver_dubins.obj.xf,solver_dubins.N)
X_interp_cartpole = line_trajectory(solver_cartpole.obj.x0,solver_cartpole.obj.xf,solver_cartpole.N)

U_pendulum = rand(solver_pendulum.model.m,solver_pendulum.N)
U_dubins = rand(solver_dubins.model.m,solver_dubins.N)
U_cartpole = rand(solver_cartpole.model.m,solver_cartpole.N)

#######################################

### Solve ###
@time results_pendulum, stats_pendulum = solve(solver_pendulum,U_pendulum)
@time results_dubins, stats_dubins = solve(solver_dubins,U_dubins)
@time results_cartpole, stats_cartpole = solve(solver_cartpole,U_cartpole)

### Results ###
println("Final state (pendulum)-> res: $(results_pendulum.X[end]), goal: $(solver_pendulum.obj.xf)\n Iterations: $(stats_pendulum["iterations"])\n Outer loop iterations: $(stats_pendulum["major iterations"])\n Max violation: $(stats_pendulum["c_max"][end])\n Max μ: $(maximum([to_array(results_pendulum.MU)[:]; results_pendulum.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_pendulum.LAMBDA)[:]; results_pendulum.λN[:]])))\n")
println("Final state (dubins)-> res: $(results_dubins.X[end]), goal: $(solver_dubins.obj.xf)\n Iterations: $(stats_dubins["iterations"])\n Outer loop iterations: $(stats_dubins["major iterations"])\n Max violation: $(stats_dubins["c_max"][end])\n Max μ: $(maximum([to_array(results_dubins.MU)[:]; results_dubins.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_dubins.LAMBDA)[:]; results_dubins.λN[:]])))\n")
println("Final state (cartpole)-> res: $(results_cartpole.X[end]), goal: $(solver_cartpole.obj.xf)\n Iterations: $(stats_cartpole["iterations"])\n Outer loop iterations: $(stats_cartpole["major iterations"])\n Max violation: $(stats_cartpole["c_max"][end])\n Max μ: $(maximum([to_array(results_cartpole.MU)[:]; results_cartpole.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_cartpole.LAMBDA)[:]; results_cartpole.λN[:]])))\n")
