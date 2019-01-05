using Random
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics","urdf")
urdf_kuka_orig = joinpath(urdf_folder, "kuka_iiwa.urdf")
urdf_kuka = joinpath(urdf_folder, "temp","kuka.urdf")

function write_kuka_urdf()
    kuka_mesh_dir = joinpath(TrajectoryOptimization.root_dir(),"dynamics","urdf","kuka_iiwa_mesh")
    temp_dir = joinpath(TrajectoryOptimization.root_dir(),"dynamics","urdf","temp")
    if !isdir(temp_dir)
        mkdir(temp_dir)
    end
    open(urdf_kuka_orig,"r") do f
        open(urdf_kuka, "w") do fnew
            for ln in eachline(f)
                pre = findfirst("<mesh filename=",ln)
                post = findlast("/>",ln)
                if !(pre isa Nothing) && !(post isa Nothing)
                    inds = pre[end]+2:post[1]-2
                    pathstr = ln[inds]
                    file = splitdir(pathstr)[2]
                    ln = ln[1:pre[end]+1] * joinpath(kuka_mesh_dir,file) * ln[post[1]-1:end]
                end
                println(fnew,ln)
            end
        end
    end
end

function get_kuka_ee(kuka)
    ee_body = findbody(kuka, "iiwa_link_ee")
    ee_point = Point3D(default_frame(ee_body),0.,0.,0.)
    return ee_body, ee_point
end


function kuka_ee_ik(kuka::Mechanism,point::Vector,ik_iterations=1000,attempts=20,tol=1e-2)
    state = MechanismState(kuka)
    world = root_frame(kuka)

    # Get end-effector
    ee_body, ee_point = get_kuka_ee(kuka)

    # Run IK
    err = Inf
    iter = 1
    while err > tol
        rand!(state)
        goal = Point3D(world,point)
        ik_res = jacobian_transpose_ik!(state,ee_body,ee_point,goal,iterations=ik_iterations)
        point_res = transform(ik_res,ee_point,world).v
        err = norm(point-point_res)
        if iter > attempts
            error("IK cannot get sufficiently close to the goal")
        end
        return ik_res
    end
end

function calc_ee_position(kuka::Mechanism,X::Vector)
    ee = zero.(X)
    N = length(X)
    state = MechanismState(kuka)
    world = root_frame(kuka)
    ee_point = get_kuka_ee(kuka)[2]
    nn = num_positions(kuka)
    for k = 1:N
        set_configuration!(state, X[k][1:nn])
        ee[k] = transform(state, ee_point, world).v
    end
    return ee
end


function jacobian_transpose_ik!(state::MechanismState,
                               body::RigidBody,
                               point::Point3D,
                               desired::Point3D;
                               α=0.1,
                               iterations=100)
    mechanism = state.mechanism
    world = root_frame(mechanism)

    # Compute the joint path from world to our target body
    p = path(mechanism, root_body(mechanism), body)
    # Allocate the point jacobian (we'll update this in-place later)
    Jp = point_jacobian(state, p, transform(state, point, world))

    q = copy(configuration(state))

    for i in 1:iterations
        # Update the position of the point
        point_in_world = transform(state, point, world)
        # Update the point's jacobian
        point_jacobian!(Jp, state, p, point_in_world)
        # Compute an update in joint coordinates using the jacobian transpose
        Δq = α * Array(Jp)' * (transform(state, desired, world) - point_in_world).v
        # Apply the update
        q .= configuration(state) .+ Δq
        set_configuration!(state, q)
    end
    state
end

# Write new urdf file with correct absolute paths
write_kuka_urdf()

model = Model(urdf_kuka)
n,m = model.n, model.m

# initial and goal states
x0 = zeros(n)
xf = zeros(n)
xf[1] = pi/2
xf[2] = pi/2

# costs
Q = 1e-4*Diagonal(I,n)
Qf = 250.0*Diagonal(I,n)
R = 1e-4*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

kuka = [model, obj_uncon]
