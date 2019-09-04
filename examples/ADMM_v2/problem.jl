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

function gen_prob(agent)
    num_lift = 3
    N = 51
    dt = 0.2

    n_lift = 13
    m_lift = 5

    n_load = 6
    m_load = 3

    mass_load = load_params.m

    goal_dist = 6.
    shift_ = zeros(n_lift)
    shift_[1:3] = [0.0;0.0;0.0]
    scaling = 1.25
    x10 = zeros(n_lift)
    x10[4] = 1.
    x10[1:3] = scaling*[sqrt(8/9);0.;4/3]
    x10 += shift_
    x20 = zeros(n_lift)
    x20[4] = 1.
    x20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
    x20 += shift_
    x30 = zeros(n_lift)
    x30[4] = 1.
    x30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
    x30 += shift_
    xload0 = zeros(n_load)
    xload0[3] = 4/6
    xload0[1:3] += shift_[1:3]

    xlift0 = [x10,x20,x30]

    _shift = zeros(n_lift)
    _shift[1:3] = [goal_dist;0.0;0.0]

    # goal state
    xloadf = zeros(n_load)
    xloadf[1:3] = xload0[1:3] + _shift[1:3]
    x1f = copy(x10) + _shift
    x2f = copy(x20) + _shift
    x3f = copy(x30) + _shift

    xliftf = [x1f,x2f,x3f]

    # midpoint desired configuration
    ℓ1 = norm(x30[1:3]-x10[1:3])
    norm(x10[1:3]-x20[1:3])
    norm(x20[1:3]-x30[1:3])

    ℓ2 = norm(xload0[1:3]-x10[1:3])
    norm(xload0[1:3]-x20[1:3])
    norm(xload0[1:3]-x30[1:3])

    ℓ3 = 0.

    _shift_ = zeros(n_lift)
    _shift_[1] = goal_dist/2.

    x1m = copy(x10)
    x1m += _shift_
    x1m[1] += ℓ3
    x3m = copy(x1m)
    x3m[1] -= ℓ1
    x3m[1] -= ℓ3
    x3m[2] = -0.01
    x2m = copy(x1m)
    x2m[1] = goal_dist/2.
    x2m[2] = 0.01
    x2m[3] = ℓ2 - 0.5*sqrt(4*ℓ2^2 - ℓ1^2 + ℓ3*2) + x20[3]

    xliftmid = [x1m,x2m,x3m]









    # Robot sizes
    r_lift = 0.275
    r_load = 0.2

    # Control limits for lift robots
    u_lim_l = -Inf*ones(m_lift)
    u_lim_u = Inf*ones(m_lift)
    u_lim_l[1:4] .= 0.
    u_lim_l[5] = 0.
    u_lim_u[1:4] .= 12.0/4.0
    x_lim_l_lift = -Inf*ones(n_lift)
    x_lim_l_lift[3] = 0.

    x_lim_l_load = -Inf*ones(n_load)
    x_lim_l_load[3] = 0.

    u_lim_l_load = -Inf*ones(m_load)
    u_lim_l_load .= 0.

    bnd1 = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)
    bnd2 = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u,x_min=x_lim_l_lift)
    bnd3 = BoundConstraint(n_load,m_load,x_min=x_lim_l_load,u_min=u_lim_l_load)
    bnd4 = BoundConstraint(n_load,m_load,x_min=x_lim_l_load)

    # Obstacles
    r_cylinder = 0.5

    _cyl = []
    push!(_cyl,(goal_dist/2.,1.,r_cylinder))
    push!(_cyl,(goal_dist/2.,-1.,r_cylinder))

    # push!(_cyl,(goal_dist/2.,1.25,r_cylinder))
    # push!(_cyl,(goal_dist/2.,-1.25,r_cylinder))

    function cI_cylinder_lift(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_lift)
        end
    end
    obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_load)
        end
    end
    obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)



    # Initial controls
    f1 = (x10[1:3] - xload0[1:3])/norm(x10[1:3] - xload0[1:3])
    f2 = (x20[1:3] - xload0[1:3])/norm(x20[1:3] - xload0[1:3])
    f3 = (x30[1:3] - xload0[1:3])/norm(x30[1:3] - xload0[1:3])
    f_mag = hcat(f1, f2, f3)\[0;0;9.81*mass_load]
    ff = [f_mag[1]*f1, f_mag[2]*f2, f_mag[3]*f3]

    thrust = 9.81*(quad_params.m + mass_load/num_lift)/4
    ulift = [[thrust;thrust;thrust;thrust;f_mag[i]] for i = 1:num_lift]
    ulift_r = [[0;0;0;0;f_mag[i]] for i = 1:num_lift]

    uload = vcat(f_mag...)

    # initial control mid
    xloadm = [goal_dist/2.; 0; xload0[3];0.;0.;0.]
    f1m = (x1m[1:3] - xloadm[1:3])/norm(x1m[1:3] - xloadm[1:3])
    f2m = (x2m[1:3] - xloadm[1:3])/norm(x2m[1:3] - xloadm[1:3])
    f3m = (x3m[1:3] - xloadm[1:3])/norm(x3m[1:3] - xloadm[1:3])
    f_magm = hcat(f1m, f2m, f3m)\[0;0;9.81*mass_load]
    ffm = [f_magm[1]*f1m, f_magm[2]*f2m, f_magm[3]*f3m]

    thrustm = 9.81*(quad_params.m + mass_load/num_lift)/4
    uliftm = [[thrustm;thrustm;thrustm;thrustm;f_mag[i]] for i = 1:num_lift]
    uliftm_r = [[0;0;0;0;f_mag[i]] for i = 1:num_lift]

    uloadm = vcat(f_mag...)

    Nmid = Int(floor(N/2))

    U0_lift = [[ulift[i] for k = 1:N-1] for i = 1:num_lift]
    U0_load = [uload for k = 1:N-1]

    Q_lift = 1.0e-1*Diagonal(ones(n_lift))
    Q_lift[1,1] = 1.0e-4
    r_control = 1.0e-3*ones(4)
    r_slack = ones(1)
    R_lift = 1.0*Diagonal([r_control;r_slack])
    Qf_lift = 100.0*Diagonal(ones(n_lift))

    Q_load = 0.0*Diagonal(ones(n_load))
    # Q_load[1,1] = 1.0e-4
    R_load = Diagonal([r_slack;r_slack;r_slack])
    Qf_load = 0.0*Diagonal(ones(n_load))


    obj_lift = [LQRObjective(Q_lift,R_lift,Qf_lift,xliftf[i],N,ulift[i]) for i = 1:num_lift]
    obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N,uload)

    Q_mid_lift = copy(Q_lift)
    for i in (1:3)
        Q_mid_lift[i,i] = 100.
    end

    Q_mid_load = copy(Q_load)
    for i in (1:3)
        Q_mid_load[i,i] = 100.
    end

    cost_mid_lift = [LQRCost(Q_mid_lift,R_lift,xliftmid[i],uliftm[i]) for i = 1:num_lift]
    cost_mid_load = LQRCost(Q_mid_load,R_load,xloadm,uloadm)

    for i = 1:num_lift
        obj_lift[i].cost[Nmid] = cost_mid_lift[i]
    end
    obj_load.cost[Nmid] = cost_mid_load

    # Constraints
    constraints_lift = []
    for i = 1:num_lift
        con = Constraints(N)
        # con[1] += bnd1
        for k = 1:N
            con[k] += bnd2 + obs_lift
        end
        push!(constraints_lift,copy(con))
    end

    constraints_load = Constraints(N)
    for k = 2:N-1
        constraints_load[k] += bnd3 + obs_load
    end
    constraints_load[N] += goal_constraint(xloadf) + bnd3 + obs_load

    # Create problems
    prob_lift = [Problem(gen_lift_model_initial(xload0,xlift0[i]),
                    obj_lift[i],
                    U0_lift[i],
                    integration=:midpoint,
                    constraints=constraints_lift[i],
                    x0=xlift0[i],
                    xf=xliftf[i],
                    N=N,
                    dt=dt)
                    for i = 1:num_lift]

    prob_load = Problem(gen_load_model_initial(xload0,xlift0),
                    obj_load,
                    U0_load,
                    integration=:midpoint,
                    constraints=constraints_load,
                    x0=xload0,
                    xf=xloadf,
                    N=N,
                    dt=dt)


    if agent ∈ [:load, 0]
        return prob_load
    else
        return prob_lift[agent]
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

function door_obstacles(r_cylinder=0.5)
    _cyl = NTuple{3,Float64}[]

    push!(_cyl,(5.0, 1.,r_cylinder))
    push!(_cyl,(5.0,-1.,r_cylinder))
    return _cyl
end


function gen_prob_all(lift_params, load_params, r0_load=[0,0,0.5]; quat=false, num_lift=3, N=51,
        integration=:midpoint, agent=:batch)::Problem{Float64, Discrete}

    # Params
    tf = 10.0  # sec
    goal_distance = 10.0  # m
    d = 1.55   # rope length (m)
    r_config = 1.2  # radius of initial configuration
    β = deg2rad(45)  # fan angle (radians)
    Nmid = convert(Int,floor(N/2))+1
    r_cylinder = 0.5

    # Constants
    n_lift = 13
    m_lift = 5
    n_load = 6
    m_load = num_lift

    # Calculated Params
    n_batch = num_lift*n_lift + n_load
    m_batch = num_lift*m_lift + m_load
    α = asin(r_config/d)

    # Params from params tuple
    load_mass = load_params.m
    lift_mass = lift_params.m
    load_radius = load_params.radius
    lift_radius = lift_params.radius

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DYNAMICS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    info = Dict{Symbol,Any}()
    if quat
        info[:quat] = [(4:7) .+ i for i in 0:n_lift:n_lift*num_lift-1]
    end
    batch_params = (lift=lift_params, load=load_params)
    model_batch = Model(batch_dynamics!, n_batch, m_batch, batch_params, info)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIAL CONDITIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Initial conditions
    rf_load = copy(r0_load)
    rf_load[1] += goal_distance
    xlift0, xload0 = get_states(r0_load, n_lift, n_load, num_lift, d, α)
    xliftf, xloadf = get_states(rf_load, n_lift, n_load, num_lift, d, α)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OBJECTIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    q_lift, r_lift, qf_lift = quad_costs(n_lift, m_lift)
    q_load, r_load, qf_load = load_costs(n_load, m_load)

    # determine static forces
    u0_lift, u0_load = calc_static_forces(xlift0, xload0, lift_params.m, load_mass, num_lift)



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MIDPOINT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # get position at midpoint
    rm_load = [goal_distance/2, 0, r0_load[3]]
    rm_lift = get_quad_locations(rm_load, d, β, num_lift, config=:doorway)

    xliftmid = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        xliftmid[i][1:3] = rm_lift[i]
        xliftmid[i][4] = 1.0
    end
    xliftmid[1][2] = -0.01
    xliftmid[1][3] =  0.01

    xloadm = zeros(n_load)
    xloadm[1:3] = rm_load

    # create objective at midpoint
    q_lift_mid = copy(q_lift)
    q_load_mid = copy(q_load)
    q_lift_mid[1:3] .= 1
    q_load_mid[1:3] .= 1

    uliftm, uloadm = calc_static_forces(xliftmid, xloadm, lift_params.m, load_mass, num_lift)



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONSTRAINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

    function cI_cylinder_lift(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_lift)
        end
    end

    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_load)
        end
    end

    # Bound constraints
    u_min_lift = [0,0,0,0,-Inf]
    u_min_load = zeros(num_lift)
    u_max_lift = ones(m_lift)*12/4
    u_max_lift[end] = Inf
    u_max_load = ones(m_load)*Inf

    # Create problem constraints
    con = Constraints(N)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CREATE PROBLEMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if agent == :batch

        # Initial and final conditions
        x0 = vcat(xlift0...,xload0)
        xf = vcat(xliftf...,xloadf)


        # objective costs
        Q = Diagonal([repeat(q_lift, num_lift); q_load])
        R = Diagonal([repeat(r_lift, num_lift); r_load])
        Qf = Diagonal([repeat(qf_lift, num_lift); qf_load])

        # Create objective
        u0 = vcat(u0_lift...,u0_load)
        obj = LQRObjective(Q,R,Qf,xf,N,u0)

        # Midpoint
        xm = vcat(xliftmid...,xloadm)
        um = vcat(uliftm...,uloadm)
        Q_mid = Diagonal([repeat(q_lift_mid, num_lift); q_load_mid])
        cost_mid = LQRCost(Q_mid,R,xm,um)
        obj.cost[Nmid] = cost_mid

        # Bound Constraints
        u_l = [repeat(u_min_lift, num_lift); u_min_load]
        u_u = [repeat(u_max_lift, num_lift); u_max_load]
        bnd = BoundConstraint(n_batch,m_batch,u_min=u_l,u_max=u_u)


        # Constraints
        cyl = Constraint{Inequality}(cI_cylinder,n_batch,m_batch,(num_lift+1)*length(_cyl),:cyl)
        dist_con = Constraint{Equality}(distance_constraint,n_batch,m_batch, num_lift, :distance)
        for_con = Constraint{Equality}(force_constraint,n_batch,m_batch, num_lift, :force)
        col_con = Constraint{Inequality}(collision_constraint,n_batch,m_batch, 3, :collision)
        goal = goal_constraint(xf)

        for k = 1:N-1
            con[k] += dist_con + for_con + bnd + col_con + cyl
        end
        con[N] +=  goal + col_con  + dist_con

        # Create problem
        prob = Problem(model_batch, obj, constraints=con,
                tf=tf, N=N, xf=xf, x0=x0,
                integration=integration)

        # Initial controls
        U0 = [u0 for k = 1:N-1]
        initial_controls!(prob, U0)

    elseif agent == :load

        # Objective
        Q_load = Diagonal(q_load)
        R_load = Diagonal(r_load)
        Qf_load = Diagonal(qf_load)
        obj_load = LQRObjective(Q_load, R_load, Qf_load, xloadf, N, u0_load)

        # Constraints
        bnd = BoundConstraint(n_load, m_load, u_min=u_min_load, u_max=u_max_load)
        obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)

        constraints_load = Constraints(N)
        for k = 2:N-1
            constraints_load[k] += bnd + obs_load
        end
        constraints_load[N] += goal_constraint(xloadf) + bnd + obs_load

        # Initial controls
        U0_load = [u0_load for k = 1:N-1]

        prob = Problem(gen_load_model_initial(xload0,xlift0),
                        obj_load,
                        U0_load,
                        integration=integration,
                        constraints=constraints_load,
                        x0=xload0,
                        xf=xloadf,
                        N=N,
                        tf=tf)

    elseif agent ∈ 1:num_lift

        obj_lift = [LQRObjective(Q_lift,R_lift,Qf_lift,xliftf[i],N,ulift[i]) for i = 1:num_lift]

    else
        error("Agent not valid")
    end

    return prob

end
