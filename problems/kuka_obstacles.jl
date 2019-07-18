# Kuka w/ obstacles
model = Dynamics.kuka_model
model_d = rk3(model)
n,m = model.n, model.m

T = Float64

kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)
kuka_state = MechanismState(kuka)
nn_kuka = num_positions(kuka_state)
world = root_frame(kuka)

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
    set_configuration!(state,x[1:nn_kuka])
    [RigidBodyDynamics.transform(state,point,world).v for point in points]
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
circles_kuka = Vector{Float64}[]
push!(circles_kuka,[d,0.0,1.2,0.2])
push!(circles_kuka,[0,-d,0.4,0.15])
push!(circles_kuka,[0,-d,1.2,0.15])

cylinders_kuka = [[d,-d,0.08],[d,d,0.08],[-d,-d,0.08]]

points,radii,frames = kuka_points(kuka,false)

x0 = zeros(n)
x0[2] = pi/2
x0[3] = pi/2
x0[4] = pi/2
xf = zeros(n)
xf[1] = pi/2
xf[4] = pi/2

state_cache = StateCache(kuka)
num_obstacles = length(circles_kuka)+length(cylinders_kuka)
c = zeros(length(points)*num_obstacles)
cI_arm_obstacles = generate_collision_constraint(kuka,circles_kuka,cylinders_kuka)


Q = Diagonal([ones(7); ones(7)*100])
Qf = 10.0*Diagonal(I,n)
R = 1e-2*Diagonal(I,m)
tf = 5.0
xf_ee = Dynamics.end_effector_function(xf)
x0_ee = Dynamics.end_effector_function(x0)

N = 41 # number of knot points
dt = tf/(N-1)

U_hold = hold_trajectory(n,m,N, kuka, x0[1:7])
obj = LQRObjective(Q,R,Qf,xf,N)

obs = Constraint{Inequality}(cI_arm_obstacles,n,m,length(points)*num_obstacles,:obs)
bnd = BoundConstraint(n,m,u_min=-80.,u_max=80.)
goal = goal_constraint(xf)
constraints = Constraints(N)
constraints[1] += bnd
for k = 2:N-1
    constraints[k] += bnd + obs
end
constraints[N] += goal

kuka_obstacles_problem = Problem(model_d, obj, x0=x0, xf=xf, N=N, dt=dt, constraints=constraints)
initial_controls!(kuka_obstacles_problem, U_hold)

kuka_obstacles_objects = (circles_kuka,cylinders_kuka)
