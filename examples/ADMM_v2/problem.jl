include("models.jl")
function gen_prob(agent)
    num_lift = 3
    N = 51
    dt = 0.2

    n_lift = 13
    m_lift = 5

    n_load = 6
    m_load = 3

    ceiling = 2.1

    goal_dist = 6.
    shift_ = zeros(n_lift)
    shift_[1:3] = [0.0;0.0;-0.25]
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
    xload0[1:3] += shift_[1:3]
    xload0[3] = 0.5


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
    x_lim_u_lift = Inf*ones(n_lift)
    x_lim_u_lift[3] = ceiling

    x_lim_l_load = -Inf*ones(n_load)
    x_lim_l_load[3] = 0.

    u_lim_l_load = -Inf*ones(m_load)
    u_lim_l_load .= 0.

    bnd1 = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u,x_max=x_lim_u_lift)
    bnd2 = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u,x_min=x_lim_l_lift,x_max=x_lim_u_lift)
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
