using TrajectoryOptimization
using Plots

n = 13 # states (quadrotor w/ quaternions)
m = 4 # controls

# Setup solver options
opts = SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache=true
opts.c1=1e-4
opts.c2=2.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-5
opts.iterations_outerloop = 250
opts.iterations = 1000

# Objective and constraints
Qf = 100.0*eye(n)
Q = 1e-2*eye(n)
R = 1e-2*eye(m)
tf = 5.0
dt = 0.1

x0 = zeros(n)
quat0 = eul2quat([0.0; pi/2; 0.0]) # ZYX Euler angles
x0[4:7] = quat0[:,1]
x0

xf = zeros(n)
xf[1:3] = [10.0;10.0;5.0] # xyz position
quatf = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
xf[4:7] = quatf
xf

u_min = -3.0
u_max = 3.0

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max)

model! = Model(Dynamics.quadrotor_dynamics!,n,m)

solver_uncon = Solver(model!,obj_uncon,dt=dt,opts=opts)
solver_con = Solver(model!,obj_con,dt=dt,opts=opts)

U = rand(solver_uncon.model.m, solver_uncon.N)

results_uncon = solve(solver_uncon,U)
results_con = solve(solver_con,U)

plot(results_uncon.X[1:3,:]',title="Quadrotor Position xyz",xlabel="Time",ylabel="Position",label=["x";"y";"z"])

plot(results_uncon.U[1:m,:]',color="green")
plot!(results_con.U[1:m,:]',color="red")

println("Final position: $(results_uncon.X[1:3,end]) | desired: $(obj_uncon.xf[1:3])")

plot_3D_trajectory(results_uncon, solver_uncon, xlim=[-1.0;11.0],ylim=[-1.0;11.0],zlim=[-1.0;11.0])
