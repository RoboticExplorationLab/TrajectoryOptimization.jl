include("models.jl")

"""
Return the 3D positions of the quads given the position of the load
Default Config:
    Distribute quads evenly around a circle centered around the load, each at a distance `d` from the load.
    The angle `α` specifies the angle between the rope and vertical (i.e. α=pi/2 puts the quads in plane with the load)
    The angle `ϕ` specifies how much the formation is rotated about Z
Doorway Config:
    Distribute quads evenly over an arc of `2α` degrees, centered at vertical, in the x-z plane
"""
function get_quad_locations(x_load::Vector, d::Real, α=π/4, num_lift=3;
        config=:default, r_cables=[zeros(3) for i = 1:num_lift], ϕ=0.0)
    if config == :default
        h = d*cos(α)
        r = d*sin(α)
        z = x_load[3] + h
        circle(θ) = [x_load[1] + r*cos(θ), x_load[2] + r*sin(θ)]
        θ = range(0,2π,length=num_lift+1) .+ ϕ
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            if num_lift == 2
                x_lift[i][1:2] = circle(θ[i] + pi/2)
            else
                x_lift[i][1:2] = circle(θ[i])
            end
            x_lift[i][3] = z
            x_lift[i] += r_cables[i]  # Shift by attachment location
        end
    elseif config == :doorway
        y = x_load[2]
        fan(θ) = [x_load[1] - d*sin(θ), y, x_load[3] + d*cos(θ)]
        θ = range(-α,α, length=num_lift)
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            x_lift[i][1:3] = fan(θ[i])
        end
    end
    return x_lift
end

function gen_prob(agent, quad_params, load_params; num_lift=3, N=51, quat=false, obs=true)


    # Params
    dt = 0.2
    # tf = 10.0  # sec
    goal_dist = 6.0  # m
    d = 1.55   # rope length (m)
    r_config = 1.2  # radius of initial configuration
    β = deg2rad(50)  # fan angle (radians)
    Nmid = convert(Int,floor(N/2))+1
    r_cylinder = 0.5
    ceiling = 2.1

    # Constants
    n_lift = 13
    m_lift = 5
    n_load = 6
    m_load = num_lift

    # Calculated Params
    n_batch = num_lift*n_lift + n_load
    m_batch = num_lift*m_lift + m_load
    α = asin(r_config/d)

    # Robot sizes
    lift_radius = 0.275
    load_radius = 0.2

    mass_load = load_params.m::Float64
    mass_lift = quad_params.m::Float64

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIAL CONDITIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Initial conditions
    r0_load = [0,0,0.25]
    rf_load = copy(r0_load)
    rf_load[1] += goal_dist
    xlift0, xload0 = get_states(r0_load, n_lift, n_load, num_lift, d, α)
    xliftf, xloadf = get_states(rf_load, n_lift, n_load, num_lift, d, α)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MIDPOINT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # midpoint desired configuration
    rm_load = [goal_dist/2, 0, r0_load[3]]
    rm_lift = get_quad_locations(rm_load, d, β, num_lift, config=:doorway)

    xliftmid = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        xliftmid[i][1:3] = rm_lift[i]
        xliftmid[i][4] = 1.0
    end
    xliftmid[2][2] = 0.01
    xliftmid[3][2] = -0.01

    xloadm = zeros(n_load)
    xloadm[1:3] = rm_load


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIAL CONTROLS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Initial controls
    ulift, uload = calc_static_forces(xlift0, xload0, quad_params.m, mass_load, num_lift)

    # initial control mid
    uliftm, uloadm = calc_static_forces(xliftmid, xloadm, quad_params.m, mass_load, num_lift)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OBJECTIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    q_lift, r_lift, qf_lift = quad_costs(n_lift, m_lift)
    q_load, r_load, qf_load = load_costs(n_load, m_load)

    # Midpoint objective
    q_lift_mid = copy(q_lift)
    q_load_mid = copy(q_load)
    q_lift_mid[1:3] .= 10
    q_load_mid[1:3] .= 10


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONSTRAINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Control limits
    u_min_lift = [0,0,0,0,-Inf]
    u_max_lift = ones(m_lift)*19/4
    u_max_lift[end] = Inf

    x_min_lift = -Inf*ones(n_lift)
    x_min_lift[3] = 0
    x_max_lift = Inf*ones(n_lift)
    x_max_lift[3] = ceiling

    u_min_load = zeros(num_lift)
    u_max_load = ones(m_load)*Inf

    x_min_load = -Inf*ones(n_load)
    x_min_load[3] = 0
    x_max_load = Inf*ones(n_load)
    x_max_load[3] = ceiling


    # Obstacles
    _cyl = door_obstacles(r_cylinder, goal_dist/2)
    function cI_cylinder_lift(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*lift_radius)
        end
    end

    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*load_radius)
        end
    end

    function distance_constraint(c,x,u=zeros(m_batch))
        xload = x[3*13 .+ (1:3)]
        c[1] = norm(x[1:3] - xload)^2 - d^2
        c[2] = norm(x[13 .+ (1:3)] - xload)^2 - d^2
        c[3] = norm(x[2*13 .+ (1:3)] - xload)^2 - d^2

        return nothing
    end

    function force_constraint(c,x,u)
        c[1] = u[5] - u[3*5 + 1]
        c[2] = u[10] - u[3*5 + 2]
        c[3] = u[15] - u[3*5 + 3]
        return nothing
    end

    function collision_constraint(c,x,u=zeros(m_batch))
        x1 = x[1:3]
        x2 = x[13 .+ (1:3)]
        x3 = x[2*13 .+ (1:3)]

        c[1] = circle_constraint(x1,x2[1],x2[2],3*lift_radius)
        c[2] = circle_constraint(x2,x3[1],x3[2],3*lift_radius)
        c[3] = circle_constraint(x3,x1[1],x1[2],3*lift_radius)

        return nothing
    end

    _cyl = door_obstacles(r_cylinder)

    function cI_cylinder(c,x,u)
        c_shift = 1
        n_slack = 3
        for p = 1:length(_cyl)
            n_shift = 0
            for i = 1:num_lift
                idx_pos = (n_shift .+ (1:13))[1:3]
                c[c_shift] = circle_constraint(x[idx_pos],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*lift_radius)
                c_shift += 1
                n_shift += 13
            end
            c[c_shift] = circle_constraint(x[3*13 .+ (1:3)],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*lift_radius)
            c_shift += 1
        end
    end



    if agent == :load

        # Objective
        Q_load = Diagonal(q_load)
        R_load = Diagonal(r_load)
        Qf_load = Diagonal(qf_load)

        obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N,uload)
        Q_mid_load = Diagonal(q_load_mid)
        cost_mid_load = LQRCost(Q_mid_load,R_load,xloadm,uloadm)
        if obs
            obj_load.cost[Nmid] = cost_mid_load
        end
        # Constraints
        obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)
        bnd_load = BoundConstraint(n_load,m_load, x_min=x_min_load, u_min=u_min_load)
        constraints_load = Constraints(N)
        for k = 2:N-1
            constraints_load[k] += bnd_load
            if obs
                constraints_load[k] += obs_load
            end
        end
        constraints_load[N] += goal_constraint(xloadf) + bnd_load
        if obs
            constraints_load[N] += obs_load
        end

        # Initial controls
        U0_load = [uload for k = 1:N-1]

        # Create problem
        prob_load = Problem(gen_load_model_initial(xload0,xlift0,load_params),
            obj_load,
            U0_load,
            integration=:midpoint,
            constraints=constraints_load,
            x0=xload0,
            xf=xloadf,
            N=N,
            dt=dt)

    elseif agent ∈ 1:num_lift

        # Objective
        Q_lift = Diagonal(q_lift)
        R_lift = Diagonal(r_lift)
        Qf_lift = Diagonal(qf_lift)
        obj_lift = [LQRObjective(Q_lift,R_lift,Qf_lift,xliftf[i],N,ulift[i]) for i = 1:num_lift]

        if obs
            Q_mid_lift = Diagonal(q_lift_mid)
            cost_mid_lift = [LQRCost(Q_mid_lift,R_lift,xliftmid[i],uliftm[i]) for i = 1:num_lift]
            for i = 1:num_lift
                obj_lift[i].cost[Nmid] = cost_mid_lift[i]
            end
        end


        # Constraints
        bnd_lift = BoundConstraint(n_lift,m_lift,u_min=u_min_lift,u_max=u_max_lift,x_min=x_min_lift,x_max=x_max_lift)
        obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

        constraints_lift = []
        for i = 1:num_lift
            con = Constraints(N)
            # con[1] += bnd1
            for k = 1:N
                con[k] += bnd_lift
                if obs
                    con[k] += obs_lift
                end
            end
            push!(constraints_lift,copy(con))
        end

        # Initial controls
        U0_lift = [[ulift[i] for k = 1:N-1] for i = 1:num_lift]

        # Create problem
        i = agent
        prob_lift = Problem(gen_lift_model_initial(xload0,xlift0[i],quad_params),
            obj_lift[i],
            U0_lift[i],
            integration=:midpoint,
            constraints=constraints_lift[i],
            x0=xlift0[i],
            xf=xliftf[i],
            N=N,
            dt=dt)

    elseif agent == :batch

        # Dynamics
        info = Dict{Symbol,Any}()
        if quat
            info[:quat] = [(4:7) .+ i for i in 0:n_lift:n_lift*num_lift-1]
        end
        batch_params = (lift=quad_params, load=load_params)
        model_batch = Model(batch_dynamics!, n_batch, m_batch, batch_params, info)

        # Initial and final conditions
        x0 = vcat(xlift0...,xload0)
        xf = vcat(xliftf...,xloadf)

        # objective costs
        Q = Diagonal([repeat(q_lift, num_lift); q_load])
        R = Diagonal([repeat(r_lift, num_lift); r_load])
        Qf = Diagonal([repeat(qf_lift, num_lift); qf_load])

        # Create objective
        u0 = vcat(ulift...,uload)
        obj = LQRObjective(Q,R,Qf,xf,N,u0)

        # Midpoint
        if obs
            xm = vcat(xliftmid...,xloadm)
            um = vcat(uliftm...,uloadm)
            Q_mid = Diagonal([repeat(q_lift_mid, num_lift); q_load_mid])
            cost_mid = LQRCost(Q_mid,R,xm,um)
            obj.cost[Nmid] = cost_mid
        end
        # Bound Constraints
        u_l = [repeat(u_min_lift, num_lift); u_min_load]
        u_u = [repeat(u_max_lift, num_lift); u_max_load]
        x_l = [repeat(x_min_lift, num_lift); x_min_load]
        x_u = [repeat(x_max_lift, num_lift); x_max_load]
        bnd = BoundConstraint(n_batch,m_batch,u_min=u_l,u_max=u_u, x_min=x_l, x_max=x_u)

        # Constraints
        cyl = Constraint{Inequality}(cI_cylinder,n_batch,m_batch,(num_lift+1)*length(_cyl),:cyl)
        dist_con = Constraint{Equality}(distance_constraint,n_batch,m_batch, num_lift, :distance)
        for_con = Constraint{Equality}(force_constraint,n_batch,m_batch, num_lift, :force)
        col_con = Constraint{Inequality}(collision_constraint,n_batch,m_batch, 3, :collision)
        goal = goal_constraint(xf)

        con = Constraints(N)
        for k = 1:N-1
            con[k] += dist_con + for_con + bnd + col_con
            if obs
                con[k] += cyl
            end
        end
        con[N] +=  goal + col_con  + dist_con
        if obs
            con[N] += cyl
        end

        # Create problem
        prob = Problem(model_batch, obj, constraints=con,
                dt=dt, N=N, xf=xf, x0=x0,
                integration=:midpoint)

        # Initial controls
        U0 = [u0 for k = 1:N-1]
        initial_controls!(prob, U0)

        prob
    end

end

function get_states(r_load, n_lift, n_load, num_lift, d=1.55, α=deg2rad(50))
    r_lift = get_quad_locations(r_load, d, α, num_lift)
    x_lift = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        x_lift[i][1:3] = r_lift[i]
        x_lift[i][4] = 1.0
    end

    x_load = zeros(n_load)
    x_load[1:3] = r_load
    return x_lift, x_load
end

function quad_costs(n_lift, m_lift)
    q_diag = 1e-1*ones(n_lift)
    q_diag[1] = 1e-3

    r_diag = 1e-3*ones(m_lift)
    r_diag[end] = 1

    qf_diag = 100*ones(n_lift)
    return q_diag, r_diag, qf_diag
end

function load_costs(n_load, m_load)
    q_diag = 0*ones(n_load)
    q_diag[1] = 1e-3
    r_diag = 1*ones(m_load)
    qf_diag = 0*ones(n_load)
    return q_diag, r_diag, qf_diag
end

function calc_static_forces(xlift::Vector{T}, xload, lift_mass, load_mass, num_lift) where T
    f1 = normalize(xlift[1][1:3] - xload[1:3])
    f2 = normalize(xlift[2][1:3] - xload[1:3])
    f3 = normalize(xlift[3][1:3] - xload[1:3])
    f_mag = hcat(f1, f2, f3)\[0;0;9.81*load_mass]
    ff = [f_mag[1]*f1, f_mag[2]*f2, f_mag[3]*f3]

    thrust = 9.81*(lift_mass + load_mass/num_lift)/4
    ulift = [[thrust; thrust; thrust; thrust; f_mag[i]] for i = 1:num_lift]
    ulift_r = [[0.;0.;0.;0.;f_mag[i]] for i = 1:num_lift]
    uload = f_mag

    return ulift, uload
end

function door_obstacles(r_cylinder=0.5, x_door=3.0)
    _cyl = NTuple{3,Float64}[]

    push!(_cyl,(x_door, 1.,r_cylinder))
    push!(_cyl,(x_door,-1.,r_cylinder))
    push!(_cyl,(x_door-0.5, 1.,r_cylinder))
    push!(_cyl,(x_door-0.5,-1.,r_cylinder))
    push!(_cyl,(x_door+0.5, 1.,r_cylinder))
    push!(_cyl,(x_door+0.5,-1.,r_cylinder))
    return _cyl
end
