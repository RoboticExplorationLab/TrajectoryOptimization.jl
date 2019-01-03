using TrajectoryOptimization
using Plots
# Pendulum
n = 2
m = 1
model, = Dynamics.pendulum # same as UDP pendulum dynamics

Qf = 30.0*Matrix(I,n,n)
Q = 0.3*Matrix(I,n,n)
R = 0.3*Matrix(I,m,m)
dt = 0.1
tf = 5.0
x0 = [0.0; 0.0]
xf = [pi; 0.0]

obj = LQRObjective(Q,R,Qf,tf,x0,xf)

opts = SolverOptions()
opts.cost_tolerance = 1e-5
opts.gradient_tolerance = 1e-5
opts.verbose = true
opts.cache = true
opts.c2 = 100.0

solver = TrajectoryOptimization.Solver(model,obj,dt=dt,opts=opts)
U = zeros(solver.model.m, solver.N)
results, stats = TrajectoryOptimization.solve(solver,U)

# Cartpole
n = 4
m = 1
model = Model(Dynamics.cartpole_dynamic_udp!,n,m)

dt = .1; # time step
# quadratic costs
Qf = 1000*Matrix(I,n,n);
Q = .1*Matrix(I,n,n)
R = .01*Matrix(I,m,m);

tf = 5.0
x0 = [0.0; 0.0; 0.0; 0.0]
xf = [0.0; pi; 0.0; 0.0]

obj = LQRObjective(Q,R,Qf,tf,x0,xf)

opts = SolverOptions()
opts.cost_tolerance = 1e-5
opts.gradient_tolerance = 1e-5
opts.verbose = true
opts.cache = true

solver = TrajectoryOptimization.Solver(model,obj,dt=dt,opts=opts)
U = zeros(solver.model.m, solver.N)
results, stats = TrajectoryOptimization.solve(solver,U)
println(results.X[:,end])
# plot(results.cost[1:results.termination_index])
