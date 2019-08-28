
using Combinatorics

function visualize_batch(vis, sol)
    load_model = sol.model.info[:load]
    lift_model = sol.model.info[:lift]
    num_lift = sol.model.info[:num_lift]
    n_lift, m_lift = lift_model.n, lift_model.m
    n_load, m_load = load_model.n, load_model.m

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

function gen_lift_problem(lift_model, load_model, num_lift; batch=true)
    n_lift, m_lift = lift_model.n, lift_model.m
    n_load, m_load = load_model.n, load_model.m
    cable_inds = m_lift-2:m_lift

    # Params
    N = 51
    tf = 10.0
    d = 1.5
    α = pi/4
    radius_lift = 0.1
    radius_load = 0.1
    integration = :rk3
    ceiling = 3.0

    # Calculated Params
    is_quad = n_lift == 13

    # Modify models
    lift_model.info[:radius] = radius_lift
    load_model.info[:radius] = radius_load
    load_model.info[:rope_length] = d
    load_model.info[:num_lift] = num_lift


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

    # Bounds
    x_min_lift = -Inf*ones(n_lift)
    x_max_lift =  Inf*ones(n_lift)
    u_min_lift = -Inf*ones(m_lift)
    u_max_lift =  Inf*ones(m_lift)

    x_min_lift[3] = 0.0
    x_max_lift[3] = ceiling

    if is_quad
        u_max_lift[1:4] .= 12.0 / 4.0
        u_min_lift[1:4] .= 0
    else
        u_max_lift[3] = 12.0
        u_min_lift[3] = 0
    end

    x_min_load = -Inf*ones(n_load)
    x_max_load =  Inf*ones(n_load)

    x_min_load[3] = 0.0
    x_max_load[3] = ceiling


    # Objective
    q_lift = ones(n_lift)*1e-1
    r_lift = ones(m_lift)*1e-1
    qf_lift = ones(n_lift)*10.0

    q_load = ones(n_load)*1e-1
    r_load = ones(m_load)*1e-1
    qf_load = ones(n_load)*10.0

    # Initial controls
    u0_lift = zeros(m_lift)
    if n_lift == 13
        u0_lift[1:4] .= (lift_params.mass + load_params.mass)/num_lift * 9.81/4 / cos(α)
    else
        u0_lift[3] = (lift_params.mass + load_params.mass)*9.81/num_lift
    end
    u0_lift[end] = -load_params.mass*9.81/num_lift


    # Build the problem(s)
    if batch

        # Dynamics
        batch_dynamics = gen_batch_dynamics(lift_model, load_model, num_lift)
        info = Dict{Symbol, Any}(:quat0=>4:7, :load=>load_model, :lift=>lift_model,
            :num_lift=>num_lift)

        n = n_lift*num_lift + n_load
        m = m_lift*num_lift
        batch_model = Model(batch_dynamics, n, m, slung_load_params, info)

        # Initial and Final conditions
        x0 = [vcat(x0_lift...); x0_load]
        xf = [vcat(xf_lift...); xf_load]

        # Objective
        q_diag = append!(repeat(q_lift, num_lift), q_load)
        r_diag = repeat(r_lift, num_lift)
        qf_diag = append!(repeat(qf_lift, num_lift), qf_load)

        Q = Diagonal(q_diag)
        R = Diagonal(r_diag)
        Qf = Diagonal(qf_diag)

        obj = LQRObjective(Q,R,Qf,xf,N)

        # Constraints
        con_direction, con_sign, con_collision =
            build_batch_constraints(lift_model, load_model, num_lift, d, radius_lift)
        constraints = Constraints(N)
        for k = 1:N-1
            constraints[k] += con_direction #+ con_collision #+ con_sign
        end

        # Initial controls
        u0 = repeat(u0_lift, num_lift)
        U0 = [u0 for k = 1:N-1]

        # Build batch problem
        prob = Problem(batch_model, obj, U0, x0=x0, xf=xf, constraints=constraints,
            N=N, tf=tf, integration=integration)

        return [prob]

    else
        # Objective
        Q_lift = Diagonal(q_lift)
        R_lift = Diagonal(r_lift)
        Qf_lift = Diagonal(qf_lift)

        Q_load = Diagonal(q_load)
        R_load = Diagonal(r_load)
        Qf_load = Diagonal(qf_load)

        obj_lift = [LQRObjective(Q_lift, R_lift, Qf_lift, xf, N) for xf in xf_lift]
        obj_load = LQRObjective(Q_load, R_load, Qf_load, xf_load, N)

        # Bound Constraints
        bnd_lift = BoundConstraint(n_lift, m_lift, x_min=x_min_lift, x_max=x_max_lift,
            u_min=u_min_lift, u_max=u_max_lift)

        bnd_load = BoundConstraint(n_load, m_load, x_min=x_min_load, x_max=x_max_load)

        constraints_lift = Constraints(N)
        constraints_load = Constraints(N)
        for k = 1:N
            constraints_lift[k] += bnd_lift
            constraints_load[k] += bnd_load
        end

        # Initial controls
        u0_load = -u0_lift[cable_inds]*num_lift

        U0_lift = [u0_lift for k = 1:N-1]
        U0_load = [u0_load for k = 1:N-1]

        # Problems
        prob_load = Problem(load_model, obj_load, U0_load, x0=x0_load, xf=xf_load,
            constraints=constraints_load, N=N, tf=tf, integration=integration)

        prob_lift = [Problem(lift_model, obj_lift[agent], U0_lift,
            x0=x0_lift[agent],
            xf=xf_lift[agent],
            constraints=constraints_lift,
            N=N,
            tf=tf,
            integration=integration)
            for agent = 1:num_lift]
        return [prob_load; prob_lift]
    end
end


function build_lift_constraints(lift_model, load_model, X_cache, U_cache, agent, d, r_lift)
    X_load = X_cache[1]
    U_load = U_cache[1]
    N = length(X_load)
    num_lift = length(X_cache)-1

    n_lift, m_lift = lift_model.n, lift_model.m
    n_load, m_load = load_model.n, load_model.m

    cable_inds = m_lift-2:m_lift

    other_agents = collect(1:num_lift)
    filter!(x->x!=agent, other_agents)

    con_cable = Vector{Constraint{Equality}}(undef, N)
    con_sign = Vector{Constraint{Equality}}(undef, N)
    for k = 1:N
        function force_direction(c,x,u)
            x_lift = x[1:3]
            x_load = X_load[k]
            dir = x_lift - x_load
            u_cable = u[cable_inds]
            c[1:3] = cross(dir, u_cable)
            c[4] = norm(dir) - d
            c[5:9] = u_cable + U_load[k]
        end
        con_cable[k] = Constraint{Equality}(force_direction, n_lift, m_lift, 9, :cable)

        function force_sign(c,x,u)
            x_lift = x[1:3]
            x_load = X_load[k]
            dir = x_lift - x_load
            u_cable = u[cable_inds]
            c[1] = dir'u_cable
        end
        con_sign[k] = Constraint{Equality}(force_sign, n_lift, m_lift, 1, :force_sign)

        function self_collision(c,x,u)
            for (k,i) in enumerate(other_agents)
                x_i = x[1:2]
                x_j = X_cache[i+1][k]
                dist = x_j - x_i
                c[k] = norm(dist) - 2r_lift
            end
        end
        con_collision[k] = Constraint{Inequality}(force_sign, n_lift, m_lift, num_lift-1, :collision)

    end
    return
end


function build_batch_constraints(lift_model, load_model, num_lift, d::Float64, r_lift::Float64)
    n_lift, m_lift = lift_model.n, lift_model.m
    n_load, m_load = load_model.n, load_model.m
    n = n_lift*num_lift + n_load
    m = m_lift*num_lift

    # Cache indices to separate agents
    x_inds = [(i-1)*n_lift .+ (1:n_lift) for i = 1:num_lift]
    push!(x_inds, num_lift*n_lift .+ (1:n_load))

    u_inds = [(i-1)*m_lift .+ (1:m_lift) for i = 1:num_lift]
    push!(u_inds, num_lift*m_lift .+ (1:m_load))

    cable_inds = m_lift-2:m_lift

    # Constraints
    function force_direction(c,x,u)
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

    return con_direction, con_sign, con_collision
end

opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=200)
opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-5,
    iterations=10,
    penalty_scaling=2.0,
    penalty_initial=10.)

# Pick Lift and Load models
num_lift = 3
lift_model = doubleintegrator3D_lift
lift_model = quadrotor_lift
load_model = Dynamics.doubleintegrator3D


probs = gen_lift_problem(lift_model, load_model, num_lift, batch=false)
res = [solve(prob, opts_al)[1] for prob in probs]

anim = visualize_lift_system(vis, res, door=:false)

@time res, = solve(prob, opts_al)
findmax_violation(res)
visualize_batch(vis,res)



# Initial controls
u0_lift = zeros(m_lift)
if n_lift == 13
    u0_lift[1:4] .= (lift_params.mass + load_params.mass)/num_lift * 9.81/4 / cos(α)
else
    u0_lift[3] = (lift_params.mass + load_params.mass)*9.81/num_lift
end
u0_lift[end] = -load_params.mass*9.81/num_lift

# Test dynamics
xdot = zeros(n)
evaluate!(xdot, batch_model, x0, u0)

# Solve
prob = Problem(rk3(batch_model), obj, U0, x0=x0, xf=xf, constraints=constraints, N=N, tf=tf)
res, = solve(prob, opts_al)
findmax_violation(res)
visualize_batch(vis,res)


visualize_batch(vis,res)
