### Solver options ###
opts = SolverOptions()
opts.verbose=false
opts.constraint_tolerance = 1e-3
opts.τ = 0.25
######################

### Pendulum ###
n = 2 # number of pendulum states
m = 1 # number of pendulum controls
model! = Model(Dynamics.pendulum_dynamics!,n,m) # inplace dynamics model
################

## Unconstrained (infeasible)
obj_uncon = Dynamics.pendulum[2]
# obj_uncon.R[:] = [1e-2]
# obj_uncon.tf = 3.0
solver_uncon = Solver(model!,obj_uncon,dt=0.1,opts=opts)

# -Initial state and control trajectories
X_interp = line_trajectory(solver_uncon.obj.x0,solver_uncon.obj.xf,solver_uncon.N)
U = ones(solver_uncon.model.m,solver_uncon.N)

results, stats = solve(solver_uncon,X_interp,U)

# Test final state from foh solve
@test norm(results.X[end] - solver_uncon.obj.xf) < 1e-3
#############################

## Constraints ##
u_min = -3
u_max = 3
x_min = [-10;-10]
x_max = [10; 10]
obj_uncon = Dynamics.pendulum[2]
obj = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

solver = Solver(model!,obj,dt=0.1,opts=opts)

# Linear interpolation for state trajectory
X_interp = line_trajectory(solver.obj.x0,solver.obj.xf,solver.N)
U = zeros(solver.model.m,solver.N)

results, = solve(solver,X_interp,U)
max_violation(results)

# Test final state from foh solve
@test norm(results.X[end] - solver.obj.xf) < 1e-3
##################

# n,m,N = get_sizes(solver)
# m̄,mm = TrajectoryOptimization.get_num_controls(solver)
# bp = TrajectoryOptimization.BackwardPassZOH(n,mm,N)
# TrajectoryOptimization._backwardpass!(results,solver,bp)
#
#
#
# for i = 1:100
#     v = TrajectoryOptimization._backwardpass!(results,solver,bp)
#     println(v)
# end
#
#
# bp
