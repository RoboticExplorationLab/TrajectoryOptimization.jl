using TrajectoryOptimization
using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO

# Visualization using MeshCat (ie, Drake Visualizer)
visualize = true

### Solver options ###
opts = SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache=true
# opts.c1=1e-4
opts.c2=2.0
opts.constraint_tolerance = 1e-3
opts.cost_intermediate_tolerance = 1e-5
opts.cost_tolerance = 1e-5
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
model_euler = Model(Dynamics.quadrotor_dynamics_euler!,12,m)


# Objective and constraints
Qf = 100.0*eye(n)
Q = (0.1)*eye(n)
R = (0.01)*eye(m)
tf = 5.0
dt = 0.05

Qf_euler = 100.0*eye(12)
Q_euler = (0.1)*eye(12)

# -initial state
x0 = zeros(n)
quat0 = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
x0[4:7] = quat0
x0

x0_euler = zeros(12)

# -final state
xf = zeros(n)
xf[1:3] = [20.0;20.0;0.0] # xyz position
quatf = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
xf[4:7] = quatf
xf

xf_euler = zeros(12)
xf_euler[1:3] = [20.0;20.0;0.0]

# -control limits
u_min = -10.0
u_max = 10.0

# -obstacles
quad_radius = 3.0
sphere_radius = 1.0

# x0[1:3] = -1.*ones(3)
# xf[1:3] = 11.0*ones(3)
# n_spheres = 3
# spheres = ([5.0;7.0;3.0],[5.0;7.0;3.0],[5.0;7;3.0],[sphere_radius;sphere_radius;sphere_radius])
# function cI(x,u)
#     [sphere_constraint(x,spheres[1][1],spheres[2][1],spheres[3][1],spheres[4][1]+quad_radius);
#      sphere_constraint(x,spheres[1][2],spheres[2][2],spheres[3][2],spheres[4][2]+quad_radius);
#      sphere_constraint(x,spheres[1][3],spheres[2][3],spheres[3][3],spheres[4][3]+quad_radius)]
# end

n_spheres = 3
spheres = ([5.0;9.0;15.0;],[5.0;9.0;15.0],[0.0;0.0;0.0],[sphere_radius;sphere_radius;sphere_radius])
function cI(x,u)
    [sphere_constraint(x,spheres[1][1],spheres[2][1],spheres[3][1],spheres[4][1]+quad_radius);
     sphere_constraint(x,spheres[1][2],spheres[2][2],spheres[3][2],spheres[4][2]+quad_radius);
     sphere_constraint(x,spheres[1][3],spheres[2][3],spheres[3][3],spheres[4][3]+quad_radius);
     -x[3]]
end

# -constraint that quaternion should be unit
function cE(x,u)
    [x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2 - 1.0]
end

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_uncon_euler = UnconstrainedObjective(Q_euler, R, Qf_euler, tf, x0_euler, xf_euler)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, cI=cI, cE = cE)
obj_con_euler = TrajectoryOptimization.ConstrainedObjective(obj_uncon_euler, u_min=u_min, u_max=u_max,cI=cI)

# Solver
solver_uncon = Solver(model!,obj_uncon,integration=:rk4,dt=dt,opts=opts)
solver = Solver(model!,obj_con,integration=:rk4,dt=dt,opts=opts)

solver_uncon_euler = Solver(model_euler,obj_uncon_euler,integration=:rk4,dt=dt,opts=opts)
solver_euler = Solver(model_euler,obj_con_euler,integration=:rk4,dt=dt,opts=opts)


# - Initial control and state trajectories
U = ones(solver.model.m, solver.N)
# X_interp = line_trajectory(solver)
X_interp = line_trajectory_quadrotor_quaternion(solver.obj.x0,solver.obj.xf,solver.N)::Array{Float64,2}
##################

### Solve ###
# results_uncon, stats_uncon = solve(solver_uncon,X_interp,U)
# results,stats = solve(solver,U)

solve_dircol(solver,X_interp,U)
# results_uncon_euler, stats_uncon_euler = solve(solver_uncon_euler,U)
# results_euler, stats_euler = solve(solver_euler,U)#results_uncon_euler.U)
#############
#
# ### Results ###
# if opts.verbose
#     println("Final position: $(results.X[1:3,end])\n       desired: $(obj_uncon.xf[1:3])\n    Iterations: $(stats["iterations"])\n Max violation: $(max_violation(results.result[results.termination_index]))")
#     println("Final position (euler): $(results_euler.X[1:3,end])\n       desired: $(obj_uncon_euler.xf[1:3])\n    Iterations: $(stats_euler["iterations"])\n Max violation: $(max_violation(results_euler.result[results.termination_index]))")
#
#     # Position
#     plot(results.X[1:3,:]',title="Quadrotor Position xyz",xlabel="Time",ylabel="Position",label=["x";"y";"z"])
#
#     # Control
#     plot(results.U[1:m,:]',color="green")
#
# end
# ###############
#
# ### Visualizer using MeshCat and GeometryTypes ###
# # Set up visualizer
# vis = Visualizer()
# open(vis)
#
# # Import quadrotor obj file
# urdf_folder = joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf")
# # urdf = joinpath(joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf"), "quadrotor.urdf")
# obj = joinpath(joinpath(Pkg.dir("TrajectoryOptimization"), "dynamics/urdf"), "quadrotor_base.obj")
#
# # color options
# green = MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0))
# red = MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0))
# blue = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
# orange = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 1.0))
# black = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
# black_transparent = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.1))
#
# # geometries
# robot_obj = load(obj)
# sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.1*quad_radius)) # trajectory points
# sphere_medium = HyperSphere(Point3f0(0), convert(Float32,quad_radius))
#
# obstacles = vis["obs"]
# traj = vis["traj"]
# target = vis["target"]
# robot = vis["robot"]
#
# # Set camera location
# settransform!(vis["/Cameras/default"], compose(Translation(25., -5., 10),LinearMap(RotZ(-pi/4))))
#
# # Create and place obstacles
# for i = 1:n_spheres
#     setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[4][i])),red)
#     settransform!(vis["obs"]["s$i"], Translation(spheres[1][i], spheres[2][i], spheres[3][i]))
# end
#
# # Create and place trajectory
# for i = 1:solver.N
#     setobject!(vis["traj"]["t$i"],sphere_small,blue)
#     settransform!(vis["traj"]["t$i"], Translation(results.X[1,i], results.X[2,i], results.X[3,i]))
# end
#
# # Create and place initial position
# setobject!(vis["robot"]["ball"],sphere_medium,black_transparent)
# setobject!(vis["robot"]["quad"],robot_obj,black)
# settransform!(vis["robot"],compose(Translation(results.X[1,1], results.X[2,1], results.X[3,1]),LinearMap(quat2rot(results.X[4:7,1]))))
#
# # Animate quadrotor
# for i = 1:solver.N
#     settransform!(vis["robot"], compose(Translation(results.X[1,i], results.X[2,i], results.X[3,i]),LinearMap(quat2rot(results.X[4:7,i]))))
#     sleep(solver.dt/2)
# end
# #####
#
#
# # Plot Euler angle trajectories
eul = zeros(3,solver.N)
for i = 1:solver.N
    # eul[:,i] = TrajectoryOptimization.quat2eul(results.X[4:7,i])
    eul[:,i] = results_euler.X[4:6,i]
end

plot(eul',title=("Euler angle trajectories"))
plot(results_euler.X[1:3,:]')
plot(results_euler.U')

# results.X[:,end]
# results_euler.X[:,end]
# plot(log.(results.cost[1:results.termination_index]))
# plot!(log.(results_euler.cost[1:results_euler.termination_index]))
#
# plot(results.X[1:3,:]')
# plot!(results_euler.X[1:3,:]')
#
# plot(results.U')
# plot!(results_euler.U')
