using MeshCatMechanisms
using LinearAlgebra
using RigidBodyDynamics
using Plots
using GeometryTypes
using CoordinateTransformations
import TrajectoryOptimization: hold_trajectory, Trajectory, total_time
import RigidBodyDynamics: transform
include("N_plots.jl")
include("../kuka_visualizer.jl")
model, obj = Dynamics.kuka
n,m = model.n, model.m
nn = m   # Number of positions

function plot_sphere(vis::MechanismVisualizer,frame::CartesianFrame3D,center,radius,mat,name="")
    geom = HyperSphere(Point3f0(center), convert(Float32,radius))
    setelement!(vis,frame,geom,mat,name)
end

function plot_cylinder(vis::MechanismVisualizer,frame::CartesianFrame3D,c1,c2,radius,mat,name="")
    geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
    setelement!(vis,frame,geom,mat,name)
end


# Create Mechanism
kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)
kuka_visuals = URDFVisuals(Dynamics.urdf_kuka)
state = MechanismState(kuka)
world = root_frame(kuka)


# Create Visualizer
vis = Visualizer()
mvis = MechanismVisualizer(kuka, kuka_visuals, vis[:base])
open(vis)


# Default solver options
opts = SolverOptions()
opts.cost_tolerance = 1e-6
opts.constraint_tolerance = 1e-5
dircol_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)


##########################################################
#            Full body Obstacle Avoidance                #
##########################################################

# Collision Bubbles
function kuka_points(kuka,plot=false)
    bodies = [3,4,5,6]
    radii = [0.1,0.12,0.09,0.09]
    ee_body,ee_point = Dynamics.get_kuka_ee(kuka)
    ee_radii = 0.05

    points = Point3D[]
    frames = CartesianFrame3D[]
    for (idx,radius) in zip(bodies,radii)
        body = findbody(kuka,"iiwa_link_$idx")
        frame = default_frame(body)
        point = Point3D(frame,0.,0.,0.)
        plot ? plot_sphere(mvis,frame,0,radius,body_collision,"body_$idx") : nothing
        # plot ? setelement!(vis,point,radius,"body_$idx") : nothing
        push!(points,point)
        push!(frames,frame)
    end
    frame = default_frame(ee_body)
    plot ? plot_sphere(mvis,frame,0,ee_radii,body_collision,"end_effector") : nothing
    # plot ? setelement!(vis,ee_point,ee_radii,"end_effector") : nothing
    push!(points,ee_point)
    push!(radii,ee_radii)
    return points, radii, frames
end

function calc_kuka_points(x::Vector{T},points) where T
    state = state_cache[T]
    set_configuration!(state,x[1:nn])
    [transform(state,point,world).v for point in points]
end

function generate_collision_constraint(kuka::Mechanism, circles, cylinders=[])
    # Specify points along the arm
    points, radii = kuka_points(kuka)
    num_points = length(points)
    nCircle = length(circles)
    nCylinder = length(cylinders)
    num_obstacles = nCircle + nCylinder

    function cI_obstacles(c,x,u)
        nn = length(u)

        # Get current world location of points along arm
        arm_points = calc_kuka_points(x[1:nn],points)

        C = reshape(view(c,1:num_points*num_obstacles),num_points,num_obstacles)
        for i = 1:nCircle
            c_obstacle = view(C,1:num_points,i)
            for (p,kuka_point) in enumerate(arm_points)
                c_obstacle[p] = sphere_constraint(circles[i],kuka_point,radii[p]+circles[i][4])
            end
        end
        for j = 1:nCylinder
            i = j + nCircle
            c_obstacle = view(C,1:num_points,i)
            for (p,kuka_point) in enumerate(arm_points)
                c_obstacle[p] = circle_constraint(cylinders[j],kuka_point,radii[p]+cylinders[j][3])
            end
        end
    end
end

function addcircles!(vis,circles)
    world = root_frame(kuka)
    for (i,circle) in enumerate(circles)
        p = Point3D(world,collect(circle[1:3]))
        setelement!(vis,p,circle[4],"obs$i")
    end
end

function addcylinders!(vis,cylinders,height=1.5)
    for (i,cyl) in enumerate(cylinders)
        plot_cylinder(vis,world,[cyl[1],cyl[2],0],[cyl[1],cyl[2],height],cyl[3],blue_,"cyl_$i")
    end
end
ee_fun = Dynamics.get_kuka_ee_postition_fun(kuka)

# Add obstacles
d = 0.25
circles2 = Vector{Float64}[]
push!(circles2,[d,0.0,1.2,0.2])
push!(circles2,[0,-d,0.4,0.15])
push!(circles2,[0,-d,1.2,0.15])
addcircles!(mvis,circles2)

cylinders = [[d,-d,0.08],[d,d,0.08],[-d,-d,0.08]]
addcylinders!(mvis,cylinders)

points,radii,frames = kuka_points(kuka,false)

# Plot Goal
plot_sphere(mvis,world,xf_ee,0.04,green_,"goal")
plot_sphere(mvis,world,x0_ee,0.04,red_,"start")

# Generate constraint function
state_cache = StateCache(kuka)
num_obstacles = length(circles2)+length(cylinders)
c = zeros(length(points)*num_obstacles)
cI_arm_obstacles = generate_collision_constraint(kuka,circles2,cylinders)
cI_arm_obstacles(c,x0,zeros(m))

# Build the Objective
x0 = zeros(n)
x0[2] = pi/2
x0[3] = pi/2
x0[4] = pi/2
xf = zeros(n)
xf[1] = pi/2
xf[4] = pi/2
Q = Diagonal([ones(7); ones(7)*100])
Qf = 10.0*Diagonal(I,n)
R = 1e-2*Diagonal(I,m)
tf = 5.0
obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
xf_ee = ee_fun(xf)
x0_ee = ee_fun(x0)

set_configuration!(mvis, x0[1:7])
set_configuration!(mvis, xf[1:7])
costfun = LQRCost(Q,R,Qf,xf)
obj_obs_arm = ConstrainedObjective(costfun,tf,x0,xf,cI=cI_arm_obstacles,u_min=-80,u_max=80)

# Solve the Problem
solver = Solver(model, obj_obs_arm, N=41)
solver.opts.verbose = true
solver.opts.penalty_scaling = 50
solver.opts.penalty_initial = 0.01
solver.opts.cost_tolerance_intermediate = 1e-2
solver.opts.cost_tolerance = 1e-4
solver.opts.constraint_tolerance = 1e-4
solver.opts.bp_reg_initial = 0
solver.opts.square_root = true
solver.opts.use_nesterov = false
solver.opts.iterations_outerloop = 50
solver.opts.iterations = 300
solver.opts.penalty_max = 1e8
solver.opts.outer_loop_update_type = :default
U0_hold = hold_trajectory(solver, kuka, x0[1:7])
res_obs, stats_obs = solve(solver,U0_hold)
stats_obs["iterations"]
stats_obs["max_mu_iteration"]
stats_obs["runtime"]
evals(solver,:f) / stats_obs["iterations"]
res = res_obs

# Visualize
set_configuration!(mvis, x0[1:7])
X_interp, U_interp = TrajectoryOptimization.interp_traj(201,5.,res_obs.X,res_obs.U)
animate_trajectory(mvis, X_interp, 0.01)
t = range(0,length=201,step=0.01)
q = [X_interp[1:7,i] for i = 1:201]
setanimation!(mvis,t,q)

stats0 = copy(stats_obs)
res0 = deepcopy(res_obs)
plot(stats0["c_max"],yscale=:log10)
plot!(stats_obs["c_max"])

plot(stats0["cost"],yscale=:log10)
plot!(stats_obs["cost"])



plot(res_obs.C[1:N-1],ylim=(0,0.1))

function plot_ghost_trajectory(urdf,K,N,α=0.5)
    mech = parse_urdf(urdf)
    visuals = URDFVisuals(urdf)
    state = MechanismState(mech)
    vis_el = visual_elements(mech, visuals)
    set_alpha!(vis_el,α)

    ks = round.(Int,range(1,N,length=K))
    for i = 1:K
        mvis2 = MechanismVisualizer(state, vis[Symbol("shadow$i")])
        MeshCatMechanisms._set_mechanism!(mvis2, vis_el)
        MeshCatMechanisms._render_state!(mvis2)
        set_configuration!(mvis2,res_obs.X[ks[i]][1:7])
    end
end

function set_alpha!(visuals::Vector{VisualElement}, α)
    for el in visuals
        c = el.color
        c_new = RGBA(red(c),green(c),blue(c),α)
        el.color = c_new
    end
end


plot_ghost_trajectory(Dynamics.urdf_kuka,10,solver.N,0.0)

# Solve Dircol
X0 = rollout(solver,U0_hold)
dircol_options = Dict("tol"=>1e-3,"constr_viol_tol"=>1e-2,"max_iter"=>20000,"mu_init"=>0.1)
X0 = line_trajectory(solver)
res_d, stats_d = solve_dircol(solver,to_array(res_obs.X),to_array(res_obs.U),options=dircol_options)
solver.opts.verbose = false
res_d, stats_d = solve_dircol(solver,X0,U0_hold,options=dircol_options)
evals(solver,:f) / stats_d["iterations"]

X_d, U_d = TrajectoryOptimization.interp_traj(201,5.,res_d.X,res_d.U)
animate_trajectory(mvis, X_d, 0.05)


settransform!(vis["/Cameras/default"], compose(Translation(-1.5, 0., 0.),LinearMap(RotX(0)*RotZ(0))))
settransform!(vis["/Cameras/default"], compose(Translation(0, 1.5, 0.),LinearMap(RotX(0)*RotZ(-pi/2))))

# new ghost plot
function plot_ghost_trajectory_2(urdf,traj_idx,α=0.5)
    mech = parse_urdf(urdf)
    visuals = URDFVisuals(urdf)
    state = MechanismState(mech)
    vis_el = visual_elements(mech, visuals)
    set_alpha!(vis_el,α)

    for i = 1:length(traj_idx)
        # if i == length(traj_idx)
        #     set_alpha!(vis_el,1.0)
        # end
        mvis2 = MechanismVisualizer(state, vis[Symbol("shadow$i")])
        MeshCatMechanisms._set_mechanism!(mvis2, vis_el)
        MeshCatMechanisms._render_state!(mvis2)
        set_configuration!(mvis2,res_obs.X[traj_idx[i]][1:7])
    end
end

traj_idx = [1;7;11;18;41]
set_configuration!(mvis, res_obs.X[traj_idx[end]][1:7])
plot_ghost_trajectory_2(Dynamics.urdf_kuka,traj_idx,0.65)
# note
