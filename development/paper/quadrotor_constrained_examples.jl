using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO

###
# vis = Visualizer()
# open(vis)

# Import quadrotor obj file
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics","urdf")
obj = joinpath(urdf_folder, "quadrotor_base.obj")

# color options
green_ = MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0))
red_ = MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0))
blue_ = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
orange_ = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 1.0))
black_ = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
black_transparent = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.1))

# geometries
robot_obj = FileIO.load(obj)
sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.1*r_quad)) # trajectory points
sphere_medium = HyperSphere(Point3f0(0), convert(Float32,r_quad))

obstacles = vis["obs"]
traj = vis["traj"]
traj_uncon = vis["traj_uncon"]
target = vis["target"]
robot = vis["robot"]

r_quad = 3.0
###

# Solver options
dt = 0.1
integration = :rk4
opts = SolverOptions()
opts.verbose = false

# Set up model, objective, solver
model, = TrajectoryOptimization.Dynamics.quadrotor
n = model.n
m = model.m

# -initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; r_quad]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;40.;r_quad] # xyz position
# quatf = eul2quat(zeros(3)) # ZYX Euler angles
xf[4:7] = q0

# -control limits
u_min = 0.
u_max = 100.0

x_max = Inf*ones(n)
x_min = -Inf*ones(n)
x_min[3] = 0.

Q = (1e-2)*Matrix(I,n,n)
R = (1.0)*Matrix(I,m,m)
Qf = (100.0)*Matrix(I,n,n)

# obstacles constraint
r_sphere = 3.0
spheres = ((0.,10.,r_sphere,r_sphere),(0.,20.,r_sphere,r_sphere),(0.,30.,r_sphere,r_sphere))
n_spheres = 3

function cI(c,x,u)
    for i = 1:n_spheres
        c[i] = sphere_constraint(x,spheres[i][1],spheres[i][2],spheres[i][3],spheres[i][4]+r_quad)
    end
    c
end

# unit quaternion constraint
function cE(c,x,u)
    c = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
end

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,x_min=x_min,x_max=x_max,cI=cI,cE=cE)

solver_uncon = Solver(model,obj_uncon,integration=integration,dt=dt,opts=opts)
solver_con = Solver(model,obj_con,integration=integration,dt=dt,opts=opts)

U0 = rand(solver_uncon.model.m, solver_uncon.N-1)
@time results_uncon, stats_uncon = solve(solver_uncon,U0)
@time results_con, stats_con = solve(solver_con,U0)


# # ################################################
# # ## Visualizer using MeshCat and GeometryTypes ##
# # ################################################
results = results_con
solver = solver_con

# Set camera location
settransform!(vis["/Cameras/default"], compose(Translation(30., 20., 15.),LinearMap(RotY(-pi/6))))

# Create and place obstacles
for i = 1:n_spheres
    setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[i][4])),red_)
    settransform!(vis["obs"]["s$i"], Translation(spheres[i][1], spheres[i][2], spheres[i][3]))
end

# Create and place trajectory
for i = 1:solver.N
    setobject!(vis["traj_uncon"]["t$i"],sphere_small,green_)
    settransform!(vis["traj_uncon"]["t$i"], Translation(results_uncon.X[i][1], results_uncon.X[i][2], results_uncon.X[i][3]))
end
for i = 1:solver.N
    setobject!(vis["traj"]["t$i"],sphere_small,blue_)
    settransform!(vis["traj"]["t$i"], Translation(results.X[i][1], results.X[i][2], results.X[i][3]))
end
#
# Create and place initial position
setobject!(vis["robot"]["ball"],sphere_medium,black_transparent)
setobject!(vis["robot"]["quad"],robot_obj,black_)
settransform!(vis["robot"],compose(Translation(results.X[1][1], results.X[1][2], results.X[1][3]),LinearMap(quat2rot(results.X[1][4:7]))))

# Animate quadrotor
for i = 1:solver.N
    settransform!(vis["robot"], compose(Translation(results.X[i][1], results.X[i][2], results.X[i][3]),LinearMap(quat2rot(results.X[i][4:7]))))
    sleep(solver.dt*2)
end
#

# # -obstacles
# r_quad = 3.0
# r_sphere = r_quad
#
# spheres = []
# n_spheres = 0
# s1 = 5
# s2 = 5
#
# x = 10
# y = 0
# for i in 1:
#     for j = range(0,stop=,length=)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# y = 25
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# y = 12.5
# x = 30
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# x = 45
# y = 0
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# y = 25
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2

# Create and place obstacles
# for i = 1:n_spheres
#     setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[i][4])),red_)
#     settransform!(vis["obs"]["s$i"], Translation(spheres[i][1], spheres[i][2], spheres[i][3]))
# end




# s1 = 7
# s2 = 5
#
# x = 15
# y = 0
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# y = 25
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# y = 12.5
# x = 30
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# x = 45
# y = 0
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# y = 25
# for i in range(0,stop=(s1-1)*r_sphere,length=s1)
#     for j = range(0,stop=(s2-1)*r_sphere,length=s2)
#         push!(spheres,(x, y + (i-1)*2*r_sphere + r_sphere, (j-1)*2*r_sphere + r_sphere,r_sphere))
#     end
# end
# n_spheres += s1*s2
#
# # Create and place obstacles
# for i = 1:n_spheres
#     setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[i][4])),red_)
#     settransform!(vis["obs"]["s$i"], Translation(spheres[i][1], spheres[i][2], spheres[i][3]))
# end
#
#
#
# # # wall 1
# # sx = 8
# # sy = 8
# # sz = 15
# # removed_spheres = zeros(Int,1)
# # limits1 = (2,7)
# # limits2 = (2,7)
# # for i = 1:sy
# #     for j = 1:sz
# #         if (i > limits1[1] && i < limits1[2]) && (j > limits2[1] && j < limits2[2])
# #             removed_spheres[1] += 1
# #         else
# #             push!(spheres,(10,(i-1)*2*r_sphere + r_sphere,(j-1)*2*r_sphere + r_sphere,r_sphere))
# #         end
# #     end
# # end
# #
# # n_spheres = sy*sz - removed_spheres[1]
# #
# # removed_spheres_2 = zeros(Int,1)
# #
# # # wall 2
# # zh = (8-1)*2*r_sphere + r_sphere
# # for i = 2:sx
# #     for j = 1:sy
# #         if (i > limits1[1] && i < limits1[2]) && (j > limits2[1] && j < limits2[2])
# #             removed_spheres_2[1] += 1
# #             println(removed_sphere_2[1])
# #         else
# #             push!(spheres,((i-1)*2*r_sphere + 10,(j-1)*2*r_sphere + r_sphere,zh,r_sphere))
# #         end
# #     end
# # end
# # removed_spheres
# # n_spheres += (sx-1)*sy - removed_spheres_2[1]
#
