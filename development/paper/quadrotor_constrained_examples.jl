using Plots
# using MeshCat
# using GeometryTypes
# using CoordinateTransformations
# using FileIO
# using MeshIO

# Solver options
dt = 0.1
integration = :rk3_foh
opts = SolverOptions()
opts.verbose = false
opts.iterations_innerloop = 500
opts.constraint_tolerance = 1e-3
opts.cost_intermediate_tolerance = 1e-3
opts.cost_tolerance = 1e-4
opts.R_infeasible = 1.0

opts_mintime = SolverOptions()
opts_mintime.verbose = true
opts_mintime.max_dt = 0.2
opts_mintime.minimum_time_dt_estimate = 0.1
opts_mintime.min_dt = 1e-3
opts_mintime.constraint_tolerance = 1e-2
opts_mintime.R_minimum_time = 1.0
opts_mintime.ρ_initial = 0
opts_mintime.τ = .5
opts_mintime.γ = 2.0
opts_mintime.outer_loop_update = :individual
opts_mintime.iterations_innerloop = 750
opts_mintime.iterations_outerloop = 100
opts_mintime.iterations = 5000

# Set up model, objective, solver
model, = TrajectoryOptimization.Dynamics.quadrotor_euler
n = model.n
m = model.m

model.f(zeros(n),zeros(n),zeros(m))

Qf = 100.0*Matrix(I,n,n)
Q = (1e-3)*Matrix(I,n,n)
R = (1e-2)*Matrix(I,m,m)
tf = 5.0

# -initial state
x0 = zeros(n)
# quat0 = eul2quat(zeros(3)) # ZYX Euler angles
# x0[4:7] = quat0

# -final state
xf = zeros(n)
xf[1:3] = [20.0;20.0;0.0] # xyz position
# quatf = eul2quat(zeros(3)) # ZYX Euler angles
# xf[4:7] = quatf

# -control limits
u_min = -10.0
u_max = 10.0

# -obstacles
r_quad = 3.0
r_sphere = 1.0

spheres = ((5.,5.,0.,r_sphere),(9.,9.,0.,r_sphere),(15.,15.,0.,r_sphere))
n_spheres = length(spheres)
function cI(c,x,u)
    for i = 1:n_spheres
        c[i] = sphere_constraint(x,spheres[i][1],spheres[i][2],spheres[i][3],spheres[i][4]+r_quad)
    end
    c
end

# # -constraint that quaternion should be unit
# function cE(c,x,u)
#     c = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
# end

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_uncon_min = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max)
obj_uncon_min = TrajectoryOptimization.update_objective(obj_uncon_min, tf=:min, c=0.0, Q = 1e-3*Diagonal(I,n), R = 1e-3*Diagonal(I,m), Qf = Diagonal(I,n)*0.0)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, cI=cI)
obj_con_min = TrajectoryOptimization.update_objective(obj_con, tf=:min, c=0.0, Q = 1e-3*Diagonal(I,n), R = 1e-3*Diagonal(I,m), Qf = Diagonal(I,n)*0.0)

# Solver
solver_uncon = Solver(model,obj_uncon,integration=integration,dt=dt,opts=opts)
solver_uncon_mintime = TrajectoryOptimization.Solver(model,obj_uncon_min,integration=integration,N=solver_uncon.N,opts=opts_mintime)
solver_con = Solver(model,obj_con,integration=integration,dt=dt,opts=opts)
solver_con_mintime = TrajectoryOptimization.Solver(model,obj_con_min,integration=integration,N=solver_uncon.N,opts=opts_mintime)

# - Initial control and state trajectories
U0 = rand(solver_uncon.model.m, solver_uncon.N)
X0 = line_trajectory(solver_uncon)
# X0[4:7,:] .= quat0

# ## FOH vs ZOH
# solver_uncon_z = Solver(model,obj_uncon,integration=:rk3,dt=dt,opts=opts)
# solver_uncon_f = Solver(model,obj_uncon,integration=:rk3_foh,dt=dt,opts=opts)
# res_f = evaluate_trajectory(solver_uncon_f, X0, U0)
# res_z = evaluate_trajectory(solver_uncon_z, X0, U0)
# rollout!(res_f, solver_uncon_f)
# rollout!(res_z, solver_uncon_z)
#
# Jf = cost(solver_uncon_f, res_f, res_f.X, res_f.U)
# Jz = cost(solver_uncon_z, res_z, res_z.X, res_z.U)
#
# calculate_jacobians!(res_f,solver_uncon_f)
# calculate_jacobians!(res_z,solver_uncon_z)
# vf = _backwardpass_foh!(res_f,solver_uncon_f)
# vz = _backwardpass_alt!(res_z,solver_uncon_z)
#
# forwardpass!(res_f,solver_uncon_f,vf)
# forwardpass!(res_z,solver_uncon_z,vz)
#
# plot(to_array(res_f.X_)[1:3,:]')
# plot!(to_array(res_f.xm)[1:3,1:end-1]')
#
# plot(to_array(res_z.X_)[1:3,:]')
# res_f.X .= res_f.X_
# res_f.U .= res_f.U_
#
# res_z.X .= res_z.X_
# res_z.U .= res_z.U_

# Solve
# @time results_uncon_f, stats_uncon_f = solve(solver_uncon_f,U0)
# @time results_uncon_z, stats_uncon_z = solve(solver_uncon_z,U0)
#
# plot(log.(stats_uncon_f["cost"]),title="Unconstrained Quadrotor",xlabel="iteration",ylabel="log(cost)",label="foh")
# plot!(log.(stats_uncon_z["cost"]),label=["zoh"])
#
# println("Final state (foh)-> res: $(results_uncon_f.X[end][1:3]), goal: $(solver_uncon_f.obj.xf[1:3])\n Iterations: $(stats_uncon_f["iterations"])\n Outer loop iterations: $(stats_uncon_f["major iterations"])\n ")
# println("Final state (zoh)-> res: $(results_uncon_z.X[end][1:3]), goal: $(solver_uncon_z.obj.xf[1:3])\n Iterations: $(stats_uncon_z["iterations"])\n Outer loop iterations: $(stats_uncon_z["major iterations"])\n ")

# @time results_uncon, stats_uncon = solve(solver_uncon,U0)
@time results_uncon_mintime, stats_uncon_mintime = solve(solver_uncon_mintime,U0)

# @time results_con, stats_con = solve(solver_con,U0)
# @time results_con_mintime, stats_con_mintime = solve(solver_con_mintime,X0,U0)

results_mintime

println("Final state (unconstrained)-> pos: $(results_uncon.X[end][1:3]), goal: $(solver_uncon.obj.xf[1:3])\n Iterations: $(stats_uncon["iterations"])\n Outer loop iterations: $(stats_uncon["major iterations"])\n ")
println("Final state (constrained)-> pos: $(results_con.X[end][1:3]), goal: $(solver_con.obj.xf[1:3])\n Iterations: $(stats_con["iterations"])\n Outer loop iterations: $(stats_con["major iterations"])\n Max violation: $(stats_con["c_max"][end])\n Max μ: $(maximum([to_array(results_con.μ)[:]; results_con.μN[:]]))\n Max abs(λ): $(maximum(abs.([to_array(results_con.λ)[:]; results_con.λN[:]])))\n")

# ## Results
# # Position
# plot(to_array(results_uncon.X)[1:3,:]',title="Quadrotor Position xyz",color=:blue,xlabel="Time",ylabel="Position",label=["x";"y";"z"])
# plot!(to_array(results_con.X)[1:3,:]',color=:green)
#
# # Control
# plot(to_array(results_uncon.U)[1:m,:]',title="Quadrotor Control",color=:blue)
# plot(to_array(results_con.U)[1:m,:]',color=:green)

################################################
## Visualizer using MeshCat and GeometryTypes ##
################################################
# results = results_con
# solver = solver_con
# vis = Visualizer()
# open(vis)
#
# # Import quadrotor obj file
# traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
# urdf_folder = joinpath(traj_folder, "dynamics/urdf")
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
# robot_obj = load(obj)
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
# settransform!(vis["robot"],compose(Translation(results.X[1][1], results.X[1][2], results.X[1][3]),LinearMap(quat2rot(eul2quat(results.X[1][4:7])))))
#
# # Animate quadrotor
# for i = 1:solver.N
#     settransform!(vis["robot"], compose(Translation(results.X[i][1], results.X[i][2], results.X[i][3]),LinearMap(quat2rot(eul2quat(results.X[i][4:7])))))
#     sleep(solver.dt/2)
# end
