using MeshCatMechanisms
using MeshCat
using RigidBodyDynamics
using GeometryTypes
using CoordinateTransformations
using TrajectoryOptimization

model = Dynamics.kuka_model
n,m = model.n, model.m

T = Float64

kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)
kuka_visuals = URDFVisuals(Dynamics.urdf_kuka)
state = MechanismState(kuka)
world = root_frame(kuka)

# Create Visualizer
# vis = Visualizer()
# mvis = MechanismVisualizer(kuka, kuka_visuals, vis[:base])
# open(vis)
# IJuliaCell(vis)

# function plot_sphere(vis::MechanismVisualizer,frame::CartesianFrame3D,center,radius,mat,name="")
#     geom = HyperSphere(Point3f0(center), convert(Float32,radius))
#     setelement!(vis,frame,geom,mat,name)
# end
#
# function plot_cylinder(vis::MechanismVisualizer,frame::CartesianFrame3D,c1,c2,radius,mat,name="")
#     geom = Cylinder(Point3f0(c1),Point3f0(c2),convert(Float32,radius))
#     setelement!(vis,frame,geom,mat,name)
# end

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

# function addcircles!(vis,circles)
#     world = root_frame(kuka)
#     for (i,circle) in enumerate(circles)
#         p = Point3D(world,collect(circle[1:3]))
#         setelement!(vis,p,circle[4],"obs$i")
#     end
# end

# function addcylinders!(vis,cylinders,height=1.5)
#     for (i,cyl) in enumerate(cylinders)
#         plot_cylinder(vis,world,[cyl[1],cyl[2],0],[cyl[1],cyl[2],height],cyl[3],blue_,"cyl_$i")
#     end
# end

function hold_trajectory(n,m,N, mech::Mechanism, q)
    state = MechanismState(mech)
    nn = num_positions(state)
    set_configuration!(state, q[1:nn])
    vd = zero(state.q)
    u0 = dynamics_bias(state)


    if length(q) > m
        throw(ArgumentError("system must be fully actuated to hold an arbitrary position ($(length(q)) should be > $m)"))
    end
    U0 = zeros(m,N)
    for k = 1:N
        U0[:,k] = u0
    end
    return U0
end

ee_fun = Dynamics.get_kuka_ee_postition_fun(kuka)

# Add obstacles
d = 0.25
circles2 = Vector{Float64}[]
push!(circles2,[d,0.0,1.2,0.2])
push!(circles2,[0,-d,0.4,0.15])
push!(circles2,[0,-d,1.2,0.15])
# addcircles!(mvis,circles2)

cylinders = [[d,-d,0.08],[d,d,0.08],[-d,-d,0.08]]
# addcylinders!(mvis,cylinders)

points,radii,frames = kuka_points(kuka,false)

# Build the Objective
x0 = zeros(n)
x0[2] = pi/2
x0[3] = pi/2
x0[4] = pi/2
xf = zeros(n)
xf[1] = pi/2
xf[4] = pi/2

state_cache = StateCache(kuka)
num_obstacles = length(circles2)+length(cylinders)
c = zeros(length(points)*num_obstacles)
cI_arm_obstacles = generate_collision_constraint(kuka,circles2,cylinders)

obs = Constraint{Inequality}(cI_arm_obstacles,n,m,num_obstacles,:obs)
bnd = BoundConstraint(n,m,u_min=-80.,u_max=80.,trim=true)
con = [obs,bnd];

Q = Diagonal([ones(7); ones(7)*100])
Qf = 10.0*Diagonal(I,n)
R = 1e-2*Diagonal(I,m)
tf = 5.0
xf_ee = Dynamics.end_effector_function(xf)
x0_ee = Dynamics.end_effector_function(x0)

# verbose=false
# opts_ilqr = iLQRSolverOptions{T}(verbose=true,iterations=300,live_plotting=:off)
#
# opts_al = AugmentedLagrangianSolverOptions{T}(verbose=true,opts_uncon=opts_ilqr,
#     iterations=20,cost_tolerance=1.0e-6,cost_tolerance_intermediate=1.0e-5,constraint_tolerance=1.0e-3,penalty_scaling=50.,penalty_initial=0.01)
#
# opts_altro = ALTROSolverOptions{T}(verbose=true,resolve_feasible_problem=false,opts_al=opts_al,R_inf=0.01);

N = 41 # number of knot points
dt = 0.01 # total time

U_hold = hold_trajectory(n,m,N, kuka, x0[1:7])
obj = LQRObjective(Q,R,Qf,xf,N) # objective with same stagewise costs

con_set = ProblemConstraints(con,N) # constraint trajectory

prob = Problem(model,obj, x0=x0, integration=:rk4, N=N, dt=dt)
initial_controls!(prob, U_hold); # initialize problem with controls
