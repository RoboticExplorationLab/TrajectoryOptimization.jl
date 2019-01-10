using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO

# Solver options
dt = 0.1
integration = :rk4
opts = SolverOptions()
opts.verbose = false

opts_mintime = SolverOptions()
opts_mintime.verbose = true
opts_mintime.max_dt = 0.2
opts_mintime.minimum_time_dt_estimate = 0.1
opts_mintime.min_dt = 1e-3
opts_mintime.constraint_tolerance = 1e-2
opts_mintime.R_minimum_time = 1.0
opts_mintime.bp_reg_initial = 0
opts_mintime.constraint_decrease_ratio = .5
opts_mintime.penalty_scaling = 2.0
opts_mintime.outer_loop_update_type = :individual
opts_mintime.iterations_innerloop = 750
opts_mintime.iterations_outerloop = 100
opts_mintime.iterations = 5000

# Set up model, objective, solver
model, = TrajectoryOptimization.Dynamics.quadrotor
n = model.n
m = model.m

model.f(zeros(n),zeros(n),zeros(m))

Qf = (100.0)*Matrix(I,n,n)
Q = (0.0)*Matrix(I,n,n)
R = (1e-2)*Matrix(I,m,m)
tf = 5.0

# -initial state
q0 = [1.;0.;0.;0.]
x0 = zeros(n)
x0[1:3] = [5.; 1.; 1.]
# quat0 = eul2quat(zeros(3)) # ZYX Euler angles
x0[4:7] = q0

# -final state
xf = zeros(n)
xf[1:3] = [5.0;30.0;1.0] # xyz position
# quatf = eul2quat(zeros(3)) # ZYX Euler angles
xf[4:7] = q0

# -control limits
u_min = 0.
u_max = 100.0

x_min = -Inf*ones(n)
x_max = Inf*ones(n)
x_min[1:3] = [0.;-100;0.]
x_max[1:3] = [20.;100;20.]

# -obstacles
r_quad = 3.0
r_sphere = 1.0

# spheres = ((5.,5.,0.,r_sphere),(9.,9.,0.,r_sphere),(15.,15.,0.,r_sphere))
# n_spheres = length(spheres)
# function cI(c,x,u)
#     for i = 1:n_spheres
#         c[i] = sphere_constraint(x,spheres[i][1],spheres[i][2],spheres[i][3],spheres[i][4]+r_quad)
#     end
#     c
# end

s1 = 10; s2 = 10
spheres = []
n_spheres = 0

# wall 1
y_pos1 = 10.
removed_spheres = zeros(Int,1)
x_lim = (2,7)
z_lim = (2,7)
for i = 1:s1
    for j = 1:s2
        if (i > x_lim[1] && i < x_lim[2]) && (j > z_lim[1] && j < z_lim[2])
            removed_spheres[1] += 1
        else
            push!(spheres,((i-1)*2*r_sphere + r_sphere,y_pos1,(j-1)*2*r_sphere + r_sphere,r_sphere))
        end
    end
end

n_spheres = s1*s2 - removed_spheres[1]

# # # wall 2
# y_pos2 = 20.
# removed_spheres = zeros(Int,1)
# x_lim = (4,9)
# z_lim = (4,9)
# for i = 1:s1
#     for j = 1:s2
#         if (i > x_lim[1] && i < x_lim[2]) && (j > z_lim[1] && j < z_lim[2])
#             removed_spheres[1] += 1
#         else
#             push!(spheres,((i-1)*2*r_sphere + r_sphere,y_pos2,(j-1)*2*r_sphere + r_sphere,r_sphere))
#         end
#     end
# end
# n_spheres += s1*s2 - removed_spheres[1]

function cI(c,x,u)
    for i = 1:n_spheres
        c[i] = sphere_constraint(x,spheres[i][1],spheres[i][2],spheres[i][3],spheres[i][4]+r_quad)
    end
    c
end


# -constraint that quaternion should be unit
function cE(c,x,u)
    c = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
end

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

# obj_uncon_min = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max)
# obj_uncon_min = TrajectoryOptimization.update_objective(obj_uncon_min, tf=:min, Q = 1e-3*Diagonal(I,n), R = 1e-3*Diagonal(I,m), Qf = Diagonal(I,n)*0.0)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max, cI=cI, cE=cE)
# obj_con_min = TrajectoryOptimization.update_objective(obj_con, tf=:min, Q = 1e-3*Diagonal(I,n), R = 1e-3*Diagonal(I,m), Qf = Diagonal(I,n)*0.0)

# Solver
solver_uncon = Solver(model,obj_uncon,integration=integration,dt=dt,opts=opts)

# solver_uncon_mintime = TrajectoryOptimization.Solver(model,obj_uncon_min,integration=integration,N=solver_uncon.N,opts=opts_mintime)
solver_con = Solver(model,obj_con,integration=integration,dt=dt,opts=opts)
# solver_con_mintime = TrajectoryOptimization.Solver(model,obj_con_min,integration=integration,N=solver_uncon.N,opts=opts_mintime)

# - Initial control and state trajectories
U0 = rand(solver_uncon.model.m, solver_uncon.N-1)
X0 = line_trajectory(solver_uncon)
X0[4:7,:] .= q0

xm1 = zeros(n)
xm1[1:3] = [5.;10.;10]
xm1[4:7] = q0

# xm2 = zeros(n)
# xm2[1:3] = [5.;20.;6.5]
# xm2[4:7] = q0

X_guess = [x0 xm1 xf]
X_interp = interp_rows(solver_uncon.N,tf,Array(X_guess))
plot(X_interp')

@time results_uncon, stats_uncon = solve(solver_uncon,U0)
# @time results_uncon_mintime, stats_uncon_mintime = solve(solver_uncon_mintime,U0)
solver_con.opts.verbose = false
solver_con.state.second_order_dual_update = false
solver_con.opts.use_second_order_dual_update = false
solver_con.opts.resolve_feasible = false
@time results_con, stats_con = solve(solver_con,X_interp,U0)
println(stats_con["iterations"])
# @time results_con_mintime, stats_con_mintime = solve(solver_con_mintime,X0,U0)

plot(to_array(results_uncon.X)[1:3,:]')
plot(to_array(results_con.X)[1:3,:]')

# println("Final state (unconstrained)-> pos: $(results_uncon.X[end][1:3]), goal: $(solver_uncon.obj.xf[1:3])\n Iterations: $(stats_uncon["iterations"])\n Outer loop iterations: $(stats_uncon["major iterations"])\n ")
# println("Final state (constrained)-> pos: $(results_con.X[end][1:3]), goal: $(solver_con.obj.xf[1:3])\n Iterations: $(stats_con["iterations"])\n Outer loop iterations: $(stats_con["major iterations"])\n Max violation: $(stats_con["c_max"][end])\n Max μ: $(maximum(to_array(results_con.μ)))")# Max abs(λ): $(maximum(abs.(to_array(results_con.λ)[:]))\n")

# ################################################
# ## Visualizer using MeshCat and GeometryTypes ##
# ################################################
# results = results_con
# solver = solver_con
# vis = Visualizer()
# open(vis)
#
# # Import quadrotor obj file
# traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
# urdf_folder = joinpath(traj_folder, "dynamics","urdf")
# obj = joinpath(urdf_folder, "quadrotor_base.obj")
#
# # color options
# green_ = MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0))
# red_ = MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0))
# blue_ = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
# orange_ = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 1.0))
# black_ = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
# black_transparent = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.1))
#
# # geometries
# robot_obj = FileIO.load(obj)
# sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.1*r_quad)) # trajectory points
# sphere_medium = HyperSphere(Point3f0(0), convert(Float32,r_quad))
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
#     setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[i][4])),red_)
#     settransform!(vis["obs"]["s$i"], Translation(spheres[i][1], spheres[i][2], spheres[i][3]))
# end
#
# # Create and place trajectory
# for i = 1:solver.N
#     setobject!(vis["traj"]["t$i"],sphere_small,blue_)
#     settransform!(vis["traj"]["t$i"], Translation(results.X[i][1], results.X[i][2], results.X[i][3]))
# end
#
# # Create and place initial position
# setobject!(vis["robot"]["ball"],sphere_medium,black_transparent)
# setobject!(vis["robot"]["quad"],robot_obj,black_)
# settransform!(vis["robot"],compose(Translation(results.X[1][1], results.X[1][2], results.X[1][3]),LinearMap(quat2rot(results.X[1][4:7]))))
#
# # Animate quadrotor
# for i = 1:solver.N
#     settransform!(vis["robot"], compose(Translation(results.X[i][1], results.X[i][2], results.X[i][3]),LinearMap(quat2rot(results.X[i][4:7]))))
#     sleep(solver.dt*2)
# end
