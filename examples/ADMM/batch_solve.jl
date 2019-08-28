using Combinatorics

function visualize_batch(vis, sol)
    x_inds = [(i-1)*n_lift .+ (1:n_lift) for i = 1:num_lift]
    push!(x_inds, num_lift*n_lift .+ (1:n_load))

    # Params
    r_lift = 0.1

    X = sol.X
    is_quad = n_lift == 13

    # Quad mesh
    if is_quad
        traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
        urdf_folder = joinpath(traj_folder, "dynamics","urdf")
        obj = joinpath(urdf_folder, "quadrotor_base.obj")

        quad_scaling = 0.085
        robot_obj = FileIO.load(obj)
        robot_obj.vertices .= robot_obj.vertices .* quad_scaling
        robot_mat = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
    else
        robot_obj = HyperSphere(Point3f0(0), convert(Float32,r_lift))
        robot_mat = MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0))
    end

    cable = Cylinder(Point3f0(0,0,0),Point3f0(0,0,d),convert(Float32,0.01))
    for i = 1:num_lift
        setobject!(vis["lift"]["$i"], robot_obj, robot_mat)
        setobject!(vis["cable"]["$i"], cable, MeshPhongMaterial(color=RGBA(1,0,0,1)))
    end
    setobject!(vis["load"], HyperSphere(Point3f0(0), Float32(0.1)), MeshPhongMaterial(color=RGBA(0,0,1,1)))

    anim = MeshCat.Animation(convert(Int,floor(1.0/sol.dt)))
    for k = 1:sol.N
        MeshCat.atframe(anim,vis,k) do frame
            x_load = X[k][x_inds[end]]
            settransform!(frame["load"], Translation(x_load[1:3]))

            for i = 1:num_lift
                x_lift = X[k][x_inds[i]]
                settransform!(frame["cable"]["$i"], cable_transform(x_lift[1:3], x_load[1:3]))
                if is_quad
                    settransform!(frame["lift"]["$i"], compose(Translation(x_lift[1:3]), LinearMap(Quat(x_lift[4:7]...))))
                else
                    settransform!(frame["lift"]["$i"], Translation(x_lift[1:3]))
                end

            end
        end
    end
    MeshCat.setanimation!(vis,anim)
    return anim
end


function gen_batch_dynamics(lift_model, load_model, num_lift)
    n_lift, m_lift = lift_model.n, lift_model.m
    n_load, m_load = load_model.n, load_model.m
    x_inds = [(i-1)*n_lift .+ (1:n_lift) for i = 1:num_lift]
    push!(x_inds, num_lift*n_lift .+ (1:n_load))

    u_inds = [(i-1)*m_lift .+ (1:m_lift) for i = 1:num_lift]
    push!(u_inds, num_lift*m_lift .+ (1:m_load))

    cable_inds = m_lift-2:m_lift

    function quad_load_dynamics(ẋ,x,u,params)
        params_quad = params.lift
        for i = 1:num_lift
            ẋ_lift = view(ẋ, x_inds[i])
            x_lift = view(x, x_inds[i])
            u_lift = view(u, u_inds[i])
            evaluate!(ẋ_lift, lift_model, x_lift, u_lift)
        end
        u_cables = [u[u_inds[i]][cable_inds] for i = 1:num_lift]

        params_load = params.load
        ẋ_load = view(ẋ, x_inds[end])
        x_load = view(x, x_inds[end])
        u_load = -sum(u_cables)/params.load.mass  # convert to acceleration

        evaluate!(ẋ_load, load_model, x_load, u_load)
    end
end

num_lift = 4
# lift_model = doubleintegrator3D_lift
lift_model = quadrotor_lift
load_model = Dynamics.doubleintegrator3D
batch_dynamics = gen_batch_dynamics(lift_model, load_model, num_lift)

info = Dict{Symbol, Any}(:quat0=>4:7)
n_lift, m_lift = lift_model.n, lift_model.m
n_load, m_load = load_model.n, load_model.m

n = n_lift*num_lift + n_load
m = m_lift*num_lift
batch_model = Model(batch_dynamics, n, m, slung_load_params, info)
is_quad = n_lift == 13



# Params
N = 51
tf = 5.0
d = 1.55
α = deg2rad(60)
r_lift = 0.2
r_load = 0.1

# Initial and final conditions
r0_load = [0.0, 0, 0]
rf_load = [5.0, 0, 0]

r0_lift = get_quad_locations(r0_load, d, α, num_lift)
rf_lift = get_quad_locations(rf_load, d, α, num_lift)

x0_load = zeros(n_load)
xf_load = zeros(n_load)
x0_lift = [zeros(n_lift) for i = 1:num_lift]
xf_lift = [zeros(n_lift) for i = 1:num_lift]

x0_load[1:3] = r0_load
xf_load[1:3] = rf_load
for i = 1:num_lift
    x0_lift[i][1:3] = r0_lift[i]
    xf_lift[i][1:3] = rf_lift[i]
    if is_quad
        x0_lift[i][4] = 1.0
        xf_lift[i][4] = 1.0
    end
end

x0 = [vcat(x0_lift...); x0_load]
xf = [vcat(xf_lift...); xf_load]

# Objective
Q = I(n)*1e-1
R = I(m)*1e-2
Qf = I(n)*100.0

obj = LQRObjective(Q,R,Qf,xf,N)

# Constraints
function force_direction(c,x,u)
    x_inds = [(i-1)*n_lift .+ (1:n_lift) for i = 1:num_lift]
    push!(x_inds, num_lift*n_lift .+ (1:n_load))

    u_inds = [(i-1)*m_lift .+ (1:m_lift) for i = 1:num_lift]
    push!(u_inds, num_lift*m_lift .+ (1:m_load))

    cable_inds = m_lift-2:m_lift

    off = 0
    for i = 1:num_lift
        x_lift = x[x_inds[i]][1:3]
        x_load = x[x_inds[end]][1:3]
        u_cable = u[u_inds[i]][cable_inds]
        dir = x_lift - x_load

        c[off .+ (1:3)] = cross(dir,u_cable)
        c[off + 4] = norm(dir) - d
        off += 4
    end
end
con_direction = Constraint{Equality}(force_direction, n, m, 4*num_lift, :cable)

function force_sign(c,x,u)
    x_inds = [(i-1)*n_lift .+ (1:n_lift) for i = 1:num_lift]
    push!(x_inds, num_lift*n_lift .+ (1:n_load))

    u_inds = [(i-1)*m_lift .+ (1:m_lift) for i = 1:num_lift]
    push!(u_inds, num_lift*m_lift .+ (1:m_load))

    cable_inds = m_lift-2:m_lift

    for i = 1:num_lift
        x_lift = x[x_inds[i]][1:3]
        x_load = x[x_inds[end]][1:3]
        u_cable = u[u_inds[i]][cable_inds]
        dir = x_lift - x_load

        c[i] = dir'u_cable
    end
end
con_sign = Constraint{Inequality}(force_sign, n, m, num_lift, :sign)

function self_collision(c,x,u)
    x_inds = [(i-1)*n_lift .+ (1:n_lift) for i = 1:num_lift]
    push!(x_inds, num_lift*n_lift .+ (1:n_load))

    pairs = combinations(collect(1:num_lift), 2)
    for (k,pair) in enumerate(pairs)
        i,j = pair
        x_i = x[x_inds[i]][1:2]
        x_j = x[x_inds[j]][1:2]
        dist = x_j - x_i
        c[k] = 2r_lift - norm(dist)
    end
end
con_collision = Constraint{Inequality}(self_collision, n, m, binomial(num_lift, 2), :collision)


constraints = Constraints(N)
for k = 1:N-1
    constraints[k] += con_direction + con_sign + con_collision
end

# Initial controls
u0_lift = zeros(m_lift)
if n_lift == 13
    u0_lift[1:4] .= (lift_params.mass + load_params.mass)/num_lift * 9.81/4 / cos(α)
else
    u0_lift[3] = (lift_params.mass + load_params.mass)*9.81/num_lift
end
u0_lift[end] = -load_params.mass*9.81/num_lift
u0 = repeat(u0_lift, num_lift)
U0 = [u0 for k = 1:N-1]

# Test dynamics
xdot = zeros(n)
evaluate!(xdot, batch_model, x0, u0)

# Solve
prob = Problem(rk3(batch_model), obj, U0, x0=x0, xf=xf, constraints=constraints, N=N, tf=tf)
@time res, = solve(prob, opts_al)
findmax_violation(res)
visualize_batch(vis,res)


visualize_batch(vis,res)
