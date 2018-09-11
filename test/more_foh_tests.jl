# Set random seed
srand(7)

### Solver Options ###
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
# opts.c1 = 1e-4
# opts.c2 = 2.0
# opts.mu_al_update = 10.0
opts.eps_constraint = 1e-5
opts.eps_intermediate = 1e-5
opts.eps = 1e-5
# opts.iterations_outerloop = 100
# opts.iterations = 1000
# opts.iterations_linesearch = 50
opts.infeasible_regularization = 1e6
opts.outer_loop_update = :default
opts.τ = 0.1
######################

### Simple Pendulum ###
obj_uncon_p = TrajectoryOptimization.Dynamics.pendulum![2]
model_p = Dynamics.pendulum![1]

## Infeasible start with constraints pendulum (foh)
u_min = -2
u_max = 2
x_min = [-10;-10]
x_max = [10; 10]

obj_con_p = ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)
solver_con = Solver(model_p,obj_con_p,integration=:rk3_foh,dt=dt,opts=opts)

# -Linear interpolation for state trajectory
X_interp = line_trajectory(solver_con.obj.x0,solver_con.obj.xf,solver_con.N)
U = ones(solver_con.model.m,solver_con.N)

results_inf, = solve(solver_con,X_interp,U)

# Test final state from foh solve
@test norm(results_inf.X[:,end] - solver_con.obj.xf) < 1e-3

# plot(results_inf.X',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
# plot(results_inf.U',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")
# println("Final state: $(results_inf.X[:,end])")
# println("Final cost: $(results_inf.cost[end])")
