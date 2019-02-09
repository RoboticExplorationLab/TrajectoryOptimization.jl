using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using Random

Random.seed!(123)

# Solver options
N = 201
integration = :rk4
opts = SolverOptions()
opts.verbose = false
opts.square_root = true
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5
opts.constraint_tolerance_intermediate = 1e-3
opts.outer_loop_update_type = :feedback

# Obstacle Avoidance
model,obj_uncon = TrajectoryOptimization.Dynamics.quadrotor
r_quad = 3.0
n = model.n
m = model.m
obj_con = TrajectoryOptimization.Dynamics.quadrotor_3obs[2]
spheres = TrajectoryOptimization.Dynamics.quadrotor_3obs[3]
n_spheres = length(spheres)

solver_uncon = Solver(model,obj_uncon,integration=integration,N=N,opts=opts)
solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)

U_hover = 0.5*9.81/4.0*ones(solver_uncon.model.m, solver_uncon.N-1)
X_hover = rollout(solver_uncon,U_hover)

@time results_uncon, stats_uncon = solve(solver_uncon,U_hover)
@time results_con, stats_con = solve(solver_con,U_hover)

# Minimum time
obj_mintime = update_objective(obj_con,tf=:min)
opts.square_root = false
opts.max_dt = 0.1
opts.min_dt = 1e-3
opts.minimum_time_dt_estimate = obj_con.tf/(N-1)
opts.constraint_tolerance = 1e-3
opts.R_minimum_time = 100
opts.constraint_decrease_ratio = .25
opts.penalty_scaling = 1e-3
opts.outer_loop_update_type = :feedback
opts.iterations = 1000
opts.iterations_outerloop = 20
opts.verbose = false
opts.live_plotting = false

solver_mintime = Solver(model,obj_mintime,integration=integration,N=N,opts=opts)
U_hover = 0.5*9.81/4.0*ones(solver_mintime.model.m, solver_mintime.N-1)
@time results_mintime, stats_mintime = solve(solver_mintime,to_array(results_con.U))

plot(stats_mintime["c_max"],yscale=:log10,title="Quadrotor Obstacle Avoidance (Minimum Time)",xlabel="iteration",ylabel="log(max constraint violation)",label="")
y = range(minimum(stats_mintime["c_max"]), stop=maximum(log.(stats_mintime["c_max"])),length=100)
for i = 1:length(stats_mintime["outer loop iteration index"])
    plt=plot!(stats_mintime["outer loop iteration index"][i]*ones(100),y,label="",color=:black,linestyle=:dash)
    display(plt)
end

@show stats_mintime["iterations"]
@show stats_mintime["outer loop iterations"]
@show stats_mintime["c_max"][end]
@show total_time(solver_mintime,results_mintime)

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
traj_mintime = vis["traj_mintime"]
target = vis["target"]
robot_mintime = vis["robot_mintime"]


## 3 obstacle visualization

# Set camera location
settransform!(vis["/Cameras/default"], compose(Translation(25., 15., 20.),LinearMap(RotY(-pi/12))))

# Create and place obstacles
for i = 1:n_spheres
    setobject!(vis["obs"]["s$i"],HyperSphere(Point3f0(0), convert(Float32,spheres[i][4])),red_)
    settransform!(vis["obs"]["s$i"], Translation(spheres[i][1], spheres[i][2], spheres[i][3]))
end

# Create and place trajectory
for i = 1:N
    setobject!(vis["traj_mintime"]["t$i"],sphere_small,green_)
    settransform!(vis["traj_mintime"]["t$i"], Translation(results_mintime.X[i][1], results_mintime.X[i][2], results_mintime.X[i][3]))
end

# Create and place initial position
setobject!(vis["robot_mintime"]["ball"],sphere_medium,green_transparent)
setobject!(vis["robot_mintime"]["quad"],robot_obj,green_)

# Animate quadrotor
for i = 1:N
    settransform!(vis["robot_mintime"], compose(Translation(results_mintime.X[i][1], results_mintime.X[i][2], results_mintime.X[i][3]),LinearMap(quat2rot(results_mintime.X[i][4:7]))))
    solver_mintime.state.minimum_time && i != N ? dt = results_mintime.U[i][m+1]^2 : dt = solver_mintime.dt
    sleep(dt*2)
end
