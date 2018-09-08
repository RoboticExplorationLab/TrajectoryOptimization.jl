using TrajectoryOptimization
using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO

urdf_folder = joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf")
urdf = joinpath(joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf"), "quadrotor.urdf")
obj = joinpath(joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf"), "quadrotor_base.obj")


### Solver options ###
opts = SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache=true
# opts.c1=1e-4
# opts.c2=3.0
# opts.mu_al_update = 10.0
opts.eps_constraint = 1e-5
opts.eps_intermediate = 1e-3
opts.eps = 1e-5
opts.outer_loop_update = :uniform
opts.τ = 0.1
# opts.iterations_outerloop = 250
# opts.iterations = 1000
######################

### Set up model, objective, solver ###
# Model
n = 13 # states (quadrotor w/ quaternions)
m = 4 # controls
model! = Model(Dynamics.quadrotor_dynamics!,n,m)


# Objective and constraints
Qf = 100.0*eye(n)
Q = 1e-1*eye(n)
R = 1e-1*eye(m)
tf = 5.0
dt = 0.1

x0 = zeros(n)
quat0 = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
x0[4:7] = quat0
x0

xf = zeros(n)
xf[1:3] = [20.0;20.0;3.0] # xyz position
quatf = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
xf[4:7] = quatf
xf

u_min = -100.0
u_max = 7.5

n_spheres = 3
quad_radius = 3.0
sphere_radius = 1.0
spheres = ([5.0;10.0;15.0],[5.0;10.0;15.0],[1.0;2.0;3.0],[sphere_radius;sphere_radius;sphere_radius])

function cI(x,u)
    [sphere_constraint(x,spheres[1][1],spheres[2][1],spheres[3][1],spheres[4][1])+quad_radius;
     sphere_constraint(x,spheres[1][2],spheres[2][2],spheres[3][2],spheres[4][2])+quad_radius;
     sphere_constraint(x,spheres[1][3],spheres[2][3],spheres[3][3],spheres[4][3])+quad_radius]
end

function cE(x,u)
    [x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2 - 1.0]
end

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, cI=cI, cE=cE)

# Solver
solver = Solver(model!,obj_con,integration=:rk4,dt=dt,opts=opts)

# - Initial control and state trajectories
U = ones(solver.model.m, solver.N)
X_interp = line_trajectory(solver)
##################

### Solve ###
results,stats = solve(solver,U)
#############

### Results ###
if opts.verbose
    println("Final position: $(results.X[1:3,end])\n       desired: $(obj_uncon.xf[1:3])\n    Iterations: $(stats["iterations"])\n Max violation: $(max_violation(results.result[results.termination_index]))")

    # Position
    plot(results.X[1:3,:]',title="Quadrotor Position xyz",xlabel="Time",ylabel="Position",label=["x";"y";"z"])

    # Control
    plot(results.U[1:m,:]',color="green")

    # # 3D trajectory
    # plot_3D_trajectory(results, solver, xlim=[-1.0;11.0],ylim=[-1.0;11.0],zlim=[-1.0;11.0])
    #
    # # xy, yz slice trajectories
    # plot(results.X[1,:],results.X[2,:],label=["x";"y"],xlabel="x axis",ylabel="y axis")
    # plot(results.X[2,:],results.X[3,:],label=["y";"z"],xlabel="y axis",ylabel="z axis")
    #
    # ## 2D plots of trajectory and obstacles (xy)
    # plot((solver.obj.x0[1],solver.obj.x0[2]),marker=(:circle,"red"),label="x0",xlim=(-1.1,11.1),ylim=(-1.1,11.1))
    # plot!((solver.obj.xf[1],solver.obj.xf[2]),marker=(:circle,"green"),label="xf")
    #
    # theta = linspace(0,2*pi,100)
    # for k = 1:n_spheres
    #     x_sphere = spheres[4][k]*cos.(theta)
    #     y_sphere = spheres[4][k]*sin.(theta)
    #     plot!(x_sphere+spheres[1][k],y_sphere+spheres[2][k],color="red",width=2,fill=(100),legend=:none)
    # end
    #
    # plot!(results.X[1,:],results.X[2,:])
    #
    # ## 2D plots of trajectory and obstacles (yz)
    # plot((solver.obj.x0[2],solver.obj.x0[3]),marker=(:circle,"red"),label="x0",xlim=(-1.1,11.1),ylim=(-1.1,11.1))
    # plot!((solver.obj.xf[2],solver.obj.xf[3]),marker=(:circle,"green"),label="xf")
    #
    # theta = linspace(0,2*pi,100)
    # for k = 1:n_spheres
    #     x_sphere = spheres[4][k]*cos.(theta)
    #     y_sphere = spheres[4][k]*sin.(theta)
    #     plot!(x_sphere+spheres[2][k],y_sphere+spheres[3][k],color="red",width=2,fill=(100),legend=:none)
    # end
    #
    # plot!(results.X[2,:],results.X[3,:])
end
###############

### Visualizer using MeshCat and GeometryTypes ###
# Set up visualizer
vis = Visualizer()
open(vis)

# color options
green = MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0))
red = MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0))
blue = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
orange = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 1.0))
black = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
black_transparent = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.1))

# geometries
robot_obj = load(obj)
sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.1*quad_radius))
sphere_medium = HyperSphere(Point3f0(0), convert(Float32,quad_radius))

obstacles = vis["obs"]
traj = vis["traj"]
target = vis["target"]
robot = vis["robot"]

# create and place obstacles
for i = 1:n_spheres
    setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[4][i])),orange)
    settransform!(vis["obs"]["s$i"], Translation(spheres[1][i], spheres[2][i], spheres[3][i]))
end

# create and place trajectory
for i = 1:solver.N
    setobject!(vis["traj"]["t$i"],sphere_small,blue)
    settransform!(vis["traj"]["t$i"], Translation(results.X[1,i], results.X[2,i], results.X[3,i]))
end

# create and place initial position
setobject!(vis["robot"]["ball"],sphere_medium,black_transparent)
setobject!(vis["robot"]["quad"],robot_obj,black)
settransform!(vis["robot"], Translation(solver.obj.x0[1], solver.obj.x0[2], solver.obj.x0[3]))

for i = 1:solver.N
    settransform!(vis["robot"], compose(Translation(results.X[1,i], results.X[2,i], results.X[3,i]),LinearMap(quat2rot(results.X[4:7,i]))))
    sleep(0.1)
end
#####

eul = zeros(3,solver.N)
for i = 1:solver.N
    eul[:,i] = rot2eul(quat2rot(results.X[4:7,i]./norm(results.X[4:7,i])))
end

plot(eul',title=("Euler angle trajectories"))

# N = solver.N
# t = linspace(0,2*pi,N)
# theta = sin.(t)
# phi = cos.(t)
# psi = rand(N)
#
# eul = [theta';phi';psi']
# eul_out = zeros(eul)
# for i = 1:solver.N
#     tmp = eul2quat(eul[:,i])
#     tmp2 = quat2rot(tmp)
#     eul_out[:,i] = rot2eul(tmp2)
# end
#
# plot(eul[3,:])
# plot!(eul_out[3,:])
