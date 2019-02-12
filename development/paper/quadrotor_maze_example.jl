using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using Random

##########
## Maze ##
##########
integration = :rk4
r_quad = 3.0 # based on size of mesh file
model,obj_uncon = TrajectoryOptimization.Dynamics.quadrotor
N = 101
tf = 5.0
q0 = [1.;0.;0.;0.]

# -initial state
x0 = zeros(model.n)
x0[1:3] = [0.; 0.; 10.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;60.; 10.] # xyz position
xf[4:7] = q0

# -control, state limits
u_min = 0.0
u_max = 50.0
x_max = Inf*ones(model.n)
x_min = -Inf*ones(model.n)

x_max[1:3] = [25.0; Inf; 20]
x_min[1:3] = [-25.0; -Inf; 0.]

Q = (1.0)*Matrix(I,model.n,model.n)
R = (1.0)*Matrix(I,model.m,model.m)
Qf = (1000.0)*Matrix(I,model.n,model.n)

# obstacle constraints
r_cylinder = 2.
cylinders = []
zh = 3
l1 = 5
l2 = 4
l3 = 5
l4 = 10

for i = range(-25,stop=-10,length=l1)
    push!(cylinders,(i, 10,r_cylinder))
end

for i = range(10,stop=25,length=l1)
    push!(cylinders,(i, 10, r_cylinder))
end

for i = range(-7.5,stop=7.5,length=l3)
    push!(cylinders,(i, 30, r_cylinder))
end

for i = range(-25,stop=-10,length=l1)
    push!(cylinders,(i, 50, r_cylinder))
end

for i = range(10,stop=25,length=l1)
    push!(cylinders,(i, 50, r_cylinder))
end

for i = range(10+2*r_cylinder,stop=50-2*r_cylinder,length=l4)
    push!(cylinders,(-25, i, r_cylinder))
end

for i = range(10+2*r_cylinder,stop=50-2*r_cylinder,length=l4)
    push!(cylinders,(25, i, r_cylinder))
end

n_cylinders = length(cylinders)

function cI(c,x,u)
    for i = 1:n_cylinders
        c[i] = circle_constraint(x,cylinders[i][1],cylinders[i][2],cylinders[i][3]+r_quad)
    end
    c
end

# unit quaternion constraint
function cE(c,x,u)
    c = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
end

opts = SolverOptions()
obj_uncon_maze = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon_maze,x_min=x_min,x_max=x_max,u_max=u_max,u_min=u_min,cE=cE,cI=cI)

solver_uncon = Solver(model,obj_uncon_maze,integration=integration,N=N,opts=opts)
solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)
solver_con.opts.square_root = true
solver_con.opts.R_infeasible = 0.01
solver_con.opts.resolve_feasible = true
solver_con.opts.cost_tolerance = 1e-4
solver_con.opts.cost_tolerance_intermediate = 1e-4
solver_con.opts.constraint_tolerance = 1e-4
solver_con.opts.constraint_tolerance_intermediate = 0.01
solver_con.opts.penalty_scaling = 10.0
solver_con.opts.penalty_initial = 1.0
solver_con.opts.outer_loop_update_type = :feedback
solver_con.opts.iterations_outerloop = 25
solver_con.opts.iterations = 500
solver_con.opts.iterations_innerloop = 300
solver_con.opts.use_penalty_burnin = false
solver_con.opts.verbose = true
solver_con.opts.live_plotting = false

# Initial control trajectory
U_hover = 0.5*9.81/4.0*ones(solver_uncon.model.m, solver_uncon.N-1)

# Initial state trajectory
X_guess = zeros(model.n,7)
X_guess[:,1] = x0
X_guess[:,7] = xf
X_guess[1:3,2:6] .= [0 -12.5 -20 -12.5 0 ;15 20 30 40 45 ;10 10 10 10 10]

X_guess[4:7,:] .= q0
X0 = TrajectoryOptimization.interp_rows(N,solver_uncon.obj.tf,X_guess)

plot(X_guess[1:3,:]')

# Unconstrained solve
@time results_uncon, stats_uncon = solve(solver_uncon,U_hover)

# Constrained solve
@time results_con, stats_con = solve(solver_con,X0,U_hover)

# Dircol solve
dircol_options = Dict("tol"=>solver_con.opts.cost_tolerance,"constr_viol_tol"=>solver_con.opts.constraint_tolerance)
@time results_dircol, stats_dircol = TrajectoryOptimization.solve_dircol(solver_con, X0, U_hover, options=dircol_options)

# Trajectory Plots
plot(to_array(results_uncon.U)',title="Quadrotor Unconstrained",xlabel="time",ylabel="control",labels="")
plot(to_array(results_uncon.X)[1:3,:]',title="Quadrotor Unconstrained",xlabel="time",ylabel="position",labels=["x";"y";"z"],legend=:topleft)

t_array = range(0,stop=solver_uncon.obj.tf,length=solver_uncon.N)
plot(t_array[1:N-1],to_array(results_uncon.U)',title="Quadrotor Maze",xlabel="time",ylabel="control",labels="")
plot(t_array,to_array(results_uncon.X)[1:3,:]',title="Quadrotor Maze",xlabel="time",ylabel="position",labels=["x";"y";"z"],legend=:topleft)

t_array = range(0,stop=solver_con.obj.tf,length=solver_con.N)
plot(t_array[1:end-1],to_array(results_con.U)',title="Quadrotor Maze",xlabel="time",ylabel="control",labels="")
plot(t_array,to_array(results_con.X)[1:3,:]',title="Quadrotor Maze",xlabel="time",ylabel="position",labels=["x";"y";"z"],legend=:topleft)
@assert max_violation(results_con) <= opts.constraint_tolerance

# Constraint convergence plot
plot(stats_con["c_max"],yscale=:log10,title="Quadrotor Maze",xlabel="iteration",ylabel="log(max constraint violation)",label="sqrt",legend=:bottomleft)

# Constrained results
@show stats_con["iterations"]
@show stats_con["outer loop iterations"]
@show stats_con["c_max"][end]
@show stats_con["cost"][end]

#################
# Visualization #
#################

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
red_transparent = MeshPhongMaterial(color=RGBA(1, 0, 0, 0.1))
blue_ = MeshPhongMaterial(color=RGBA(0, 0, 1, 1.0))
blue_transparent = MeshPhongMaterial(color=RGBA(0, 0, 1, 0.1))
blue_semi = MeshPhongMaterial(color=RGBA(0, 0, 1, 0.5))

orange_ = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 1.0))
orange_transparent = MeshPhongMaterial(color=RGBA(233/255, 164/255, 16/255, 0.1))
black_ = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
black_transparent = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.1))
black_semi = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.5))

function plot_cylinder(c1,c2,radius,mat,name="")
    geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
    setobject!(vis["cyl"][name],geom,red_)
end

function addcylinders!(vis,cylinders,height=1.5)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder([cyl[1],cyl[2],0],[0,0,height],cyl[3],blue_,"cyl_$i")
    end
end

# geometries
robot_obj = FileIO.load(obj)

sphere_small = HyperSphere(Point3f0(0), convert(Float32,0.1*r_quad)) # trajectory points
sphere_medium = HyperSphere(Point3f0(0), convert(Float32,0.25*r_quad))

obstacles = vis["obs"]
traj = vis["traj"]
traj_x0 = vis["traj_x0"]
traj_uncon = vis["traj_uncon"]
target = vis["target"]
robot = vis["robot"]
robot_uncon = vis["robot_uncon"]

# Set camera location
settransform!(vis["/Cameras/default"], compose(Translation(0., 75., 50.),LinearMap(RotX(pi/10)*RotZ(pi/2))))
# settransform!(vis["/Cameras/default"], compose(Translation(0., 35., 65.),LinearMap(RotX(pi/3)*RotZ(pi/2))))

# Create and place obstacles
addcylinders!(vis,cylinders,16.0)

# Create and place trajectory
# for i = 1:N
#     setobject!(vis["traj_uncon"]["t$i"],sphere_small,blue_)
#     settransform!(vis["traj_uncon"]["t$i"], Translation(results_uncon.X[i][1], results_uncon.X[i][2], results_uncon.X[i][3]))
# end

# for i = 1:N
#     setobject!(vis["traj_x0"]["t$i"],sphere_small,blue_)
#     settransform!(vis["traj_x0"]["t$i"], Translation(X0[1,i], X0[2,i], X0[3,i]))
# end
for i = 1:size(X_guess,2)
    setobject!(vis["traj_x0"]["t$i"],sphere_medium,blue_semi)
        settransform!(vis["traj_x0"]["t$i"], Translation(X_guess[1,i], X_guess[2,i], X_guess[3,i]))
end

for i = 1:N
    setobject!(vis["traj"]["t$i"],sphere_small,green_)
    settransform!(vis["traj"]["t$i"], Translation(results_con.X[i][1], results_con.X[i][2], results_con.X[i][3]))
end

# setobject!(vis["robot_uncon"]["ball"],sphere_medium,orange_transparent)
# setobject!(vis["robot_uncon"]["quad"],robot_obj,black_)

# setobject!(vis["robot"]["ball"],sphere_medium,green_transparent)
setobject!(vis["robot"]["quad"],robot_obj,black_)

# Animate quadrotor
# for i = 1:N
#     settransform!(vis["robot_uncon"], compose(Translation(results_uncon.X[i][1], results_uncon.X[i][2], results_uncon.X[i][3]),LinearMap(quat2rot(results_uncon.X[i][4:7]))))
#     sleep(solver_uncon.dt)
# end

for i = 1:N
    settransform!(vis["robot"], compose(Translation(results_con.X[i][1], results_con.X[i][2], results_con.X[i][3]),LinearMap(quat2rot(results_con.X[i][4:7]))))
    sleep(solver_con.dt)
end

i = N
settransform!(vis["robot"], compose(Translation(results_con.X[i][1], results_con.X[i][2], results_con.X[i][3]),LinearMap(quat2rot(results_con.X[i][4:7]))))

# Ghose quadrotor scene
traj_idx = [1;12;20;30;40;50;N]
n_robots = length(traj_idx)
for i = 1:n_robots
    robot = vis["robot_$i"]
    setobject!(vis["robot_$i"]["quad"],robot_obj,black_semi)
    settransform!(vis["robot_$i"], compose(Translation(results_con.X[traj_idx[i]][1], results_con.X[traj_idx[i]][2], results_con.X[traj_idx[i]][3]),LinearMap(quat2rot(results_con.X[traj_idx[i]][4:7]))))
end
