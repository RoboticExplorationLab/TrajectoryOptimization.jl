using TrajectoryOptimization
using Plots
using BenchmarkTools
using Base.Test
using Juno

n = 2 # number of pendulum states
m = 1 # number of pendulum controls
model! = Model(Dynamics.pendulum_dynamics!,n,m) # inplace dynamics model
obj_uncon = Dynamics.pendulum[2]

opts = SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
opts.cache=true
opts.c1=1e-4
opts.c2=2.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1000.0
opts.eps_constraint = 1e-2

# Constraints
u_min = -100
u_max = 100
x_min = [-100;-100]
x_max = [100; 100]
obj_uncon = Dynamics.pendulum[2]
obj = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

solver = Solver(model!,obj,dt=0.1,opts=opts)

U = ones(m,solver.N-1)
X = ones(n,solver.N)

X_interp = line_trajectory(solver.obj.x0,solver.obj.xf,solver.N)
@time results = solve_al(solver,X_interp,U)

plot(results.X',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
plot(results.U',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")
trajectory_animation(results)
trajectory_animation(results,traj="control",filename="control.gif")
# a = ConstrainedResults(0,0,0,0,0)
# size(a.X,1)
