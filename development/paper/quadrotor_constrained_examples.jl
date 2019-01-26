using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using Random

Random.seed!(123)

r_quad = 3.0

###
vis = Visualizer()
open(vis)

# Import quadrotor obj file
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics","urdf")
obj = joinpath(urdf_folder, "quadrotor_base.obj")

# color options
green_ = MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0))
green_transparent = MeshPhongMaterial(color=RGBA(0, 1, 0, 0.1))
red_ = MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0))
blue_ = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
blue_transparent = MeshPhongMaterial(color=RGBA(0, 0, 1, 0.1))

orange_ = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 1.0))
orange_transparent = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 0.1))
black_ = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
black_transparent = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.1))

# geometries
robot_obj = FileIO.load(obj)
sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.1*r_quad)) # trajectory points
sphere_medium = HyperSphere(Point3f0(0), convert(Float32,r_quad))

obstacles = vis["obs"]
traj = vis["traj"]
traj_uncon = vis["traj_uncon"]
traj_mintime = vis["traj_mintime"]
target = vis["target"]
robot = vis["robot"]
robot_uncon = vis["robot_uncon"]
robot_mintime = vis["robot_mintime"]

###

# Solver options
N = 201
integration = :rk4
opts = SolverOptions()
opts.verbose = false
opts.square_root = true
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5
opts.use_penalty_burnin = false
opts.outer_loop_update_type = :feedback

# Set up model, objective, solver
model,obj_uncon = TrajectoryOptimization.Dynamics.quadrotor
obj_con = TrajectoryOptimization.Dynamics.quadrotor_3obs[2]
spheres = TrajectoryOptimization.Dynamics.quadrotor_3obs[3]
n_spheres = length(spheres)

solver_uncon = Solver(model,obj_uncon,integration=integration,N=N,opts=opts)
solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)

U_hover = 0.5*9.81/4.0*ones(solver_uncon.model.m, solver_uncon.N-1)
X_hover = rollout(solver_uncon,U_hover)

@time results_uncon, stats_uncon = solve(solver_uncon,U_hover)
@time results_con, stats_con = solve(solver_con,U_hover)

plot(to_array(results_con.U)[:,1:solver_con.N-1]')
plot(to_array(results_con.X)[1:3,:]')
max_violation(results_con) < opts.constraint_tolerance

obj_mintime = update_objective(obj_con,tf=:min, u_min=u_min, u_max=u_max)

# opts.max_dt = 0.2
# opts.min_dt = 1e-3
# opts.minimum_time_dt_estimate = tf/(N-1)
# opts.constraint_tolerance = 0.001
# opts.R_minimum_time = 1.0
# opts.constraint_decrease_ratio = .25
# opts.penalty_scaling = 3.0
# opts.outer_loop_update_type = :individual
# opts.iterations = 1000
# opts.iterations_outerloop = 30 # 20

# opts.max_dt = 0.2
# opts.min_dt = 1e-3
# opts.minimum_time_dt_estimate = .05#tf/(N-1)
# opts.constraint_tolerance = 0.001
# opts.R_minimum_time = 1.0
# opts.constraint_decrease_ratio = .25
# opts.penalty_scaling = 3.0
# opts.outer_loop_update_type = :individual
# opts.iterations = 1000
# opts.iterations_outerloop = 30 # 20
#
# solver_mintime = Solver(model,obj_mintime,integration=integration,N=N,opts=opts)
#
# @time results_mintime, stats_mintime = solve(solver_mintime,to_array(results_con.U))
#
# plot(to_array(results_mintime.U)[:,1:solver_con.N-1]')
# plot(to_array(results_mintime.X)[1:3,1:solver_mintime.N]')
# max_violation(results_mintime)
# total_time(solver_mintime,results_mintime)

# ################################################
# ## Visualizer using MeshCat and GeometryTypes ##
# ################################################

# Set camera location
settransform!(vis["/Cameras/default"], compose(Translation(25., 15., 20.),LinearMap(RotY(-pi/12))))

# Create and place obstacles
for i = 1:n_spheres
    setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[i][4])),red_)
    settransform!(vis["obs"]["s$i"], Translation(spheres[i][1], spheres[i][2], spheres[i][3]))
end

# Create and place trajectory
for i = 1:N
    setobject!(vis["traj_uncon"]["t$i"],sphere_small,orange_)
    settransform!(vis["traj_uncon"]["t$i"], Translation(results_uncon.X[i][1], results_uncon.X[i][2], results_uncon.X[i][3]))
end
for i = 1:N
    setobject!(vis["traj"]["t$i"],sphere_small,blue_)
    settransform!(vis["traj"]["t$i"], Translation(results_con.X[i][1], results_con.X[i][2], results_con.X[i][3]))
end
# for i = 1:N
#     setobject!(vis["traj_mintime"]["t$i"],sphere_small,green_)
#     settransform!(vis["traj_mintime"]["t$i"], Translation(results_mintime.X[i][1], results_mintime.X[i][2], results_mintime.X[i][3]))
# end
#
# Create and place initial position
setobject!(vis["robot_uncon"]["ball"],sphere_medium,orange_transparent)
setobject!(vis["robot_uncon"]["quad"],robot_obj,orange_)

setobject!(vis["robot"]["ball"],sphere_medium,blue_transparent)
setobject!(vis["robot"]["quad"],robot_obj,blue_)

setobject!(vis["robot_mintime"]["ball"],sphere_medium,green_transparent)
setobject!(vis["robot_mintime"]["quad"],robot_obj,green_)

# Animate quadrotor
for i = 1:N
    settransform!(vis["robot_uncon"], compose(Translation(results_uncon.X[i][1], results_uncon.X[i][2], results_uncon.X[i][3]),LinearMap(quat2rot(results_uncon.X[i][4:7]))))
    sleep(solver_uncon.dt)
end

for i = 1:N
    settransform!(vis["robot"], compose(Translation(results_con.X[i][1], results_con.X[i][2], results_con.X[i][3]),LinearMap(quat2rot(results_con.X[i][4:7]))))
    sleep(solver_con.dt*2)
end

# for i = 1:N
#     settransform!(vis["robot_mintime"], compose(Translation(results_mintime.X[i][1], results_mintime.X[i][2], results_mintime.X[i][3]),LinearMap(quat2rot(results_mintime.X[i][4:7]))))
#     solver_mintime.state.minimum_time && i != N ? dt = results_mintime.U[i][m+1]^2 : dt = solver_con.dt
#     sleep(dt)
# end

##########
## Maze ##
##########
N = 201
tf = 5.0
q0 = [1.;0.;0.;0.]
# -initial state
x0 = zeros(model.n)
x0[1:3] = [0.; 0.; 3.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0


# -final state
xf = copy(x0)
xf[1:3] = [0.;60.;3.] # xyz position
xf[4:7] = q0

# -control limits
u_min = 0.0
u_max = 25.0
x_max = Inf*ones(model.n)
x_min = -Inf*ones(model.n)
x_max[1:3] = [25.0; Inf; 6.]
x_min[1:3] = [-25.0; -Inf; 0.]

# Q = (1e-1)*Matrix(I,model.n,model.n)
# Q[4,4] = 1.0
# Q[5,5] = 1.0
# Q[6,6] = 1.0
# Q[7,7] = 1.0
# R = (1.0)*Matrix(I,model.m,model.m)
# Qf = (100.0)*Matrix(I,model.n,model.n)
Q = (1e-1)*Matrix(I,model.n,model.n)

R = (1e-2)*Matrix(I,model.m,model.m)
Qf = (1000.0)*Matrix(I,model.n,model.n)

# obstacles constraint
# -obstacles
r_sphere = 2.
spheres = []
zh = 2
l1 = 10
l2 = 3
l3 = 20
l4 = 10

for i = range(-25,stop=-10,length=l1)
    for j = range(0,stop=zh,length=l2)
        push!(spheres,(i, 10, j*2*r_sphere + r_sphere,r_sphere))
    end
end

for i = range(10,stop=25,length=l1)
    for j = range(0,stop=zh,length=l2)
        push!(spheres,(i, 10, j*2*r_sphere + r_sphere,r_sphere))
    end
end

for i = range(-12.5,stop=12.5,length=l3)
    for j = range(0,stop=zh,length=l2)
        push!(spheres,(i, 30, j*2*r_sphere + r_sphere,r_sphere))
    end
end

for i = range(-25,stop=-10,length=l1)
    for j = range(0,stop=zh,length=l2)
        push!(spheres,(i, 50, j*2*r_sphere + r_sphere,r_sphere))
    end
end

for i = range(10,stop=25,length=l1)
    for j = range(0,stop=zh,length=l2)
        push!(spheres,(i, 50, j*2*r_sphere + r_sphere,r_sphere))
    end
end

for i = range(10+2*r_sphere,stop=50-2*r_sphere,length=l4)
    for j = range(0,stop=zh,length=l2)
        push!(spheres,(-25, i, j*2*r_sphere + r_sphere,r_sphere))
    end
end

for i = range(10+2*r_sphere,stop=50-2*r_sphere,length=l4)
    for j = range(0,stop=zh,length=l2)
        push!(spheres,(25, i, j*2*r_sphere + r_sphere,r_sphere))
    end
end
n_spheres = length(spheres)

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

opts = SolverOptions()
obj_uncon_maze = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon_maze,x_min=x_min,x_max=x_max,u_min=u_min,u_max=u_max,cI=cI,cE=cE)
solver_uncon = Solver(model,obj_uncon_maze,integration=integration,N=N,opts=opts)
solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)

# Initial control trajectory
U_hover = 0.5*9.81/4.0*ones(solver_uncon.model.m, solver_uncon.N-1)

# Initial infeasible state trajectory
X_guess = zeros(model.n,9)
X_guess[:,1] = x0
X_guess[:,9] = xf
X_guess[1:3,2:8] .= [0 -12.5 -20 -20 -20 -12.5 0; 10 20 20 30 40 40 50; 3 3 3 3 3 3 3]
X_guess[4:7,:] .= q0
X0 = TrajectoryOptimization.interp_rows(N,solver_uncon.obj.tf,X_guess)

plot(X0[1:3,:]')
@time results_uncon, stats_uncon = solve(solver_uncon,U_hover)
plot(to_array(results_uncon.U)')

U_line = to_array(results_uncon.U)

solver_con.opts.square_root = true
solver_con.opts.R_infeasible = 1.0
solver_con.opts.resolve_feasible = false
solver_con.opts.cost_tolerance = 1e-6
solver_con.opts.cost_tolerance_intermediate = 1e-4
solver_con.opts.constraint_tolerance = 1e-5
solver_con.opts.constraint_tolerance_intermediate = 0.01
solver_con.opts.penalty_scaling = 10.0
solver_con.opts.penalty_initial = 1.0
solver_con.opts.outer_loop_update_type = :feedback
solver_con.opts.iterations_outerloop = 50
solver_con.opts.iterations = 500
solver_con.opts.iterations_innerloop = 250
solver_con.opts.use_penalty_burnin = false
solver_con.opts.verbose = false
solver_con.opts.live_plotting = false
results_con, stats_con = solve(solver_con,X0,U_hover)

# solver_con.opts.R_infeasible = 1e-1
# solver_con.opts.penalty_initial = 10.0
# solver_con.opts.iterations = 250
# solver_con.opts.verbose = false
# solver_con.opts.resolve_feasible = false
# @time results_con, stats_con = solve(solver_con,X0,U_hover)
plot(to_array(results_con.U)[:,1:solver_con.N-1]')
plot(to_array(results_con.X)[1:3,:]')
max_violation(results_con)
total_time(solver_con,results_con)

# obj_mintime = update_objective(obj_con,tf=:min, u_min=u_min, u_max=u_max)

# Set camera location
settransform!(vis["/Cameras/default"], compose(Translation(0., 75., 50.),LinearMap(RotX(pi/10)*RotZ(pi/2))))


# Create and place obstacles
for i = 1:n_spheres
    setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[i][4])),red_)
    settransform!(vis["obs"]["s$i"], Translation(spheres[i][1], spheres[i][2], spheres[i][3]))
end

# Create and place trajectory
for i = 1:N
    setobject!(vis["traj_uncon"]["t$i"],sphere_small,orange_)
    settransform!(vis["traj_uncon"]["t$i"], Translation(results_uncon.X[i][1], results_uncon.X[i][2], results_uncon.X[i][3]))
end
for i = 1:N
    setobject!(vis["traj"]["t$i"],sphere_small,blue_)
    settransform!(vis["traj"]["t$i"], Translation(results_con.X[i][1], results_con.X[i][2], results_con.X[i][3]))
end
# for i = 1:N
#     setobject!(vis["traj_mintime"]["t$i"],sphere_small,green_)
#     settransform!(vis["traj_mintime"]["t$i"], Translation(results_mintime.X[i][1], results_mintime.X[i][2], results_mintime.X[i][3]))
# end
#
# Create and place initial position
setobject!(vis["robot_uncon"]["ball"],sphere_medium,orange_transparent)
setobject!(vis["robot_uncon"]["quad"],robot_obj,orange_)

setobject!(vis["robot"]["ball"],sphere_medium,blue_transparent)
setobject!(vis["robot"]["quad"],robot_obj,blue_)

setobject!(vis["robot_mintime"]["ball"],sphere_medium,green_transparent)
setobject!(vis["robot_mintime"]["quad"],robot_obj,green_)

# Animate quadrotor
for i = 1:N
    settransform!(vis["robot_uncon"], compose(Translation(results_uncon.X[i][1], results_uncon.X[i][2], results_uncon.X[i][3]),LinearMap(quat2rot(results_uncon.X[i][4:7]))))
    sleep(solver_uncon.dt)
end

for i = 1:N
    settransform!(vis["robot"], compose(Translation(results_con.X[i][1], results_con.X[i][2], results_con.X[i][3]),LinearMap(quat2rot(results_con.X[i][4:7]))))
    sleep(solver_con.dt)
end

# for i = 1:N
#     settransform!(vis["robot_mintime"], compose(Translation(results_mintime.X[i][1], results_mintime.X[i][2], results_mintime.X[i][3]),LinearMap(quat2rot(results_mintime.X[i][4:7]))))
#     solver_mintime.state.minimum_time && i != N ? dt = results_mintime.U[i][m+1]^2 : dt = solver_con.dt
#     sleep(dt)
# end
#
#
# plot(X0[1:3,:]')
#
# for i = 1:N
#     setobject!(vis["traj_uncon"]["t$i"],sphere_small,orange_)
#     settransform!(vis["traj_uncon"]["t$i"], Translation(X0[1,i], X0[2,i], X0[3,i]))
# end


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
