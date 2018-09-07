using TrajectoryOptimization
using Plots

n = 13 # states (quadrotor w/ quaternions)
m = 4 # controls

# Setup solver options
opts = SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache=true
# opts.c1=1e-4
# opts.c2=3.0
# opts.mu_al_update = 10.0
opts.eps_constraint = 1e-3
opts.eps_intermediate = 1e-3
opts.eps = 1e-3
opts.outer_loop_update = :uniform
opts.τ = 0.1
# opts.iterations_outerloop = 250
# opts.iterations = 1000

# Objective and constraints
Qf = 100.0*eye(n)
Q = 1e-1*eye(n)
R = 1e-2*eye(m)
tf = 5.0
dt = 0.1

x0 = -1*ones(n)
quat0 = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
x0[4:7] = quat0[:,1]
x0

xf = zeros(n)
xf[1:3] = [11.0;11.0;11.0] # xyz position
quatf = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
xf[4:7] = quatf
xf

u_min = -300.0
u_max = 300.0

n_spheres = 3
spheres = ([2.5;5;7.5],[2;4;7],[2.6;2.35;7.5],[0.5;0.5;0.5])
function cI(x,u)
    [sphere_constraint(x,spheres[1][1],spheres[2][1],spheres[3][1],spheres[4][1]);
     sphere_constraint(x,spheres[1][2],spheres[2][2],spheres[3][2],spheres[4][2]);
     sphere_constraint(x,spheres[1][3],spheres[2][3],spheres[3][3],spheres[4][3])]
end

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, cI=cI)

model! = Model(Dynamics.quadrotor_dynamics!,n,m)

solver = Solver(model!,obj_con,integration=:rk3,dt=dt,opts=opts)

U = ones(solver.model.m, solver.N)
X_interp = line_trajectory(solver)

results,stats = solve(solver,U)

println("Final position: $(results.X[1:3,end])\n       desired: $(obj_uncon.xf[1:3])\n    Iterations: $(stats["iterations"])\n Max violation: $(max_violation(results.result[results.termination_index]))")

plot(results.X[1:3,:]',title="Quadrotor Position xyz",xlabel="Time",ylabel="Position",label=["x";"y";"z"])

plot(results.U[1:m,:]',color="green")

plot_3D_trajectory(results, solver, xlim=[-1.0;11.0],ylim=[-1.0;11.0],zlim=[-1.0;11.0])

plot(results.X[1,:],results.X[2,:],label=["x";"y"],xlabel="x axis",ylabel="y axis")
plot(results.X[2,:],results.X[3,:],label=["y";"z"],xlabel="y axis",ylabel="z axis")

##2D plots of trajectory and obstacles (xy)
plot((solver.obj.x0[1],solver.obj.x0[2]),marker=(:circle,"red"),label="x0",xlim=(-1.1,11.1),ylim=(-1.1,11.1))
plot!((solver.obj.xf[1],solver.obj.xf[2]),marker=(:circle,"green"),label="xf")

theta = linspace(0,2*pi,100)
for k = 1:n_spheres
    x_sphere = spheres[4][k]*cos.(theta)
    y_sphere = spheres[4][k]*sin.(theta)
    plot!(x_sphere+spheres[1][k],y_sphere+spheres[2][k],color="red",width=2,fill=(100),legend=:none)
end

plot!(results.X[1,:],results.X[2,:])


##2D plots of trajectory and obstacles (yz)
plot((solver.obj.x0[2],solver.obj.x0[3]),marker=(:circle,"red"),label="x0",xlim=(-1.1,11.1),ylim=(-1.1,11.1))
plot!((solver.obj.xf[2],solver.obj.xf[3]),marker=(:circle,"green"),label="xf")

theta = linspace(0,2*pi,100)
for k = 1:n_spheres
    x_sphere = spheres[4][k]*cos.(theta)
    y_sphere = spheres[4][k]*sin.(theta)
    plot!(x_sphere+spheres[2][k],y_sphere+spheres[3][k],color="red",width=2,fill=(100),legend=:none)
end

plot!(results.X[2,:],results.X[3,:])
