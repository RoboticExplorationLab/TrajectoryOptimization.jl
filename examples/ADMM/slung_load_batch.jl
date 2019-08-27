


function slung_load_dynamics(ẋ,x,u,params)
    ẋ[1:3] = x[4:6]
    ẋ[4:6] = params.gravity + u[1:3] + u[4:6]/params.lift.mass
    ẋ[7:9] = x[10:12]
    ẋ[10:12] = params.gravity - u[4:6]/params.load.mass
end

n = 6+6
m = 3+3
slung_load_model = Model(slung_load_dynamics, n, m, slung_load_params)

# Params
N = 51
tf = 10.0

# Initial and final conditions
x0 = zeros(n)
x0[3] = 1
x0[10] = -0.5

xf = copy(x0)
shift = [3,0,0]
xf[1:3] += shift
xf[7:9] += shift

cable_length = x0[3] - x0[9]

# Objective
Q = I(n)*1e-1
R = I(m)*1e-1
Qf = I(n)*100.0

obj = LQRObjective(Q,R,Qf,xf,N)

# Constraints
function force_direction(c,x,u)
    x_lift = x[1:3]
    x_load = x[7:9]
    u_cable = u[4:6]
    dir = x_lift - x_load
    c[1:3] = cross(dir,u_cable)
    c[4] = norm(dir) - 1
end
con_direction = Constraint{Equality}(force_direction, n, m, 4, :cable)

function force_sign(c,x,u)
    x_lift = x[1:3]
    x_load = x[7:9]
    u_cable = u[4:6]
    dir = x_lift - x_load
    c[1] = dir'u_cable
end
con_sign = Constraint{Inequality}(force_sign, n, m, 1, :sign)


constraints = Constraints(N)
for k = 1:N-1
    constraints[k] += con_direction + con_sign
end

# Initial controls
u0 = zeros(m)
u0[3] = (lift_params.mass + load_params.mass)*9.81
u0[6] = -load_params.mass*9.81
U0 = [u0 for k = 1:N-1]

prob = Problem(rk3(slung_load_model), obj, U0, x0=x0, xf=xf, constraints=constraints, N=N, tf=tf)
res, = solve(prob, opts_al)
visualize_slung_load(vis, res)

f = [u[4:6] for u in res.U]
d = [x[1:3] - x[7:9] for x in res.X[1:end-1]]
plot([f'normalize(d) for (f,d) in zip(f,d)])
plot(res.X,1:3)
plot(res.U,4:6)


vis = Visualizer()
open(vis)
function visualize_slung_load(vis, sol)
    x0 = sol.x0
    d = x0[3] - x0[9]
    X = sol.X
    setobject!(vis["lift"], HyperSphere(Point3f0(0), Float32(0.2)), MeshPhongMaterial(color=RGBA(0,0,1,1)))
    setobject!(vis["load"], HyperSphere(Point3f0(0), Float32(0.1)), MeshPhongMaterial(color=RGBA(0,0,1,1)))

    cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d),convert(Float32,0.01))
    setobject!(vis["cable"], cable, MeshPhongMaterial(color=RGBA(1,0,0,1)))

    anim = MeshCat.Animation(convert(Int,floor(1.0/sol.dt)))
    for k = 1:sol.N
        MeshCat.atframe(anim,vis,k) do frame
            x_lift = X[k][1:3]
            x_load = X[k][7:9]
            settransform!(frame["cable"], cable_transform(x_lift,x_load))
            # settransform!(frame["cable"], Translation(0,0,0))
            settransform!(frame["lift"], Translation(x_lift))
            settransform!(frame["load"], Translation(x_load))
        end
    end
    MeshCat.setanimation!(vis,anim)
    return anim
end
visualize_slung_load(vis, res)


#########################################################
#                 QUADROTOR LIFT                        #
#########################################################

function quad_load_dynamics(ẋ,x,u,params)
    params_quad = params.lift
    ẋ_quad = view(ẋ,1:13)
    x_quad = view(x,1:13)
    u_quad = u

    params_load = params.load
    ẋ_load = view(ẋ,14:19)
    x_load = view(x,14:19)
    u_load = -view(u,5:7)/params_load.mass # convert to acceleration

    quadrotor_lift_dynamics!(ẋ_quad, x_quad, u_quad, params_quad)
    Dynamics.double_integrator_3D_dynamics!(ẋ_load, x_load, u_load)
end

n = 13+6
m = 4+3
info = Dict{Symbol, Any}(:quat=>4:7)
quad_load_model = Model(quad_load_dynamics, n, m, slung_load_params, info)

# Params
N = 51
tf = 5.0

# Initial and final conditions
x0 = zeros(n)
x0[3] = 1
x0[4] = 1.0  # quaternion

xf = copy(x0)
shift = [3,0,0]
xf[1:3] += shift
xf[14:16] += shift

cable_length = x0[3] - x0[16]

# Objective
Q = I(n)*1e-1
R = I(m)*1e-1
Qf = I(n)*100.0

obj = LQRObjective(Q,R,Qf,xf,N)

# Constraints
function force_direction(c,x,u)
    x_lift = x[1:3]
    x_load = x[14:16]
    u_cable = u[5:7]
    dir = x_lift - x_load
    c[1:3] = cross(dir,u_cable)
    c[4] = norm(dir) - 1
end
con_direction = Constraint{Equality}(force_direction, n, m, 4, :cable)

function force_sign(c,x,u)
    x_lift = x[1:3]
    x_load = x[14:16]
    u_cable = u[5:7]
    dir = x_lift - x_load
    c[1] = dir'u_cable
end
con_sign = Constraint{Inequality}(force_sign, n, m, 1, :sign)


constraints = Constraints(N)
for k = 1:N-1
    constraints[k] += con_direction + con_sign
end

# Initial controls
u0 = zeros(m)
u0[1:4] .= (lift_params.mass + load_params.mass)*9.81/4
u0[7] = -load_params.mass*9.81
U0 = [u0 for k = 1:N-1]

# Solve
prob = Problem(rk3(quad_load_model), obj, U0, x0=x0, xf=xf, constraints=constraints, N=N, tf=tf)
res, = solve(prob, opts_al)
visualize_quad_load(vis, res)

plot(res.U,5:7)

function visualize_quad_load(vis, sol)
    x0 = sol.x0
    d = x0[3] - x0[16]
    X = sol.X

    # Quad mesh
    traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
    urdf_folder = joinpath(traj_folder, "dynamics","urdf")
    obj = joinpath(urdf_folder, "quadrotor_base.obj")

    quad_scaling = 0.085
    robot_obj = FileIO.load(obj)
    robot_obj.vertices .= robot_obj.vertices .* quad_scaling
    robot_mat = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))

    setobject!(vis["lift"], robot_obj, robot_mat)
    setobject!(vis["load"], HyperSphere(Point3f0(0), Float32(0.1)), MeshPhongMaterial(color=RGBA(0,0,1,1)))

    cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d),convert(Float32,0.01))
    setobject!(vis["cable"], cable, MeshPhongMaterial(color=RGBA(1,0,0,1)))

    anim = MeshCat.Animation(convert(Int,floor(1.0/sol.dt)))
    for k = 1:sol.N
        MeshCat.atframe(anim,vis,k) do frame
            x_lift = X[k][1:3]
            x_load = X[k][14:16]
            settransform!(frame["cable"], cable_transform(x_lift,x_load))
            settransform!(frame["lift"], compose(Translation(x_lift), LinearMap(Quat(X[k][4:7]...))))
            settransform!(frame["load"], Translation(x_load))
        end
    end
    MeshCat.setanimation!(vis,anim)
    return anim
end
visualize_quad_load(vis, res)
