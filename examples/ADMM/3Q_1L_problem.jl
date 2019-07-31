include("methods.jl")
include("models.jl")

function build_quad_problem(agent)
    num_lift = 3

    n_lift = quadrotor_lift.n
    m_lift = quadrotor_lift.m

    n_load = doubleintegrator3D_load.n
    m_load = doubleintegrator3D_load.m

    #~~~~~~~~~~~~~~~~~~ CONSTRAINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Robot sizes
    r_lift = 0.2
    r_load = 0.1

    # Control limits for lift robots
    u_lim_l = -Inf*ones(m_lift)
    u_lim_u = Inf*ones(m_lift)
    u_lim_l[1:4] .= 0.
    u_lim_u[1:4] .= 9.81*(quad_params.m + 1.)/4.0
    bnd = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)

    # Obstacles
    r_cylinder = 0.75

    _cyl = []
    push!(_cyl,(5.,1.3,r_cylinder))
    push!(_cyl,(5.,-1.3,r_cylinder))

    function cI_cylinder_lift(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 2*r_lift)
        end
    end
    obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 2*r_load)
        end
    end
    obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)



    #~~~~~~~~~~~~~~~~~~ INITIAL CONDITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    shift_ = zeros(n_lift)
    shift_[1:3] = [0.0;0.0;1.0]
    scaling = 1.
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

    xlift0 = [x10,x20,x30]

    _shift = zeros(n_lift)
    _shift[1:3] = [10.0;0.0;0.0]

    # goal state
    xloadf = zeros(n_load)
    xloadf[1:3] = xload0[1:3] + _shift[1:3]
    x1f = copy(x10) + _shift
    x2f = copy(x20) + _shift
    x3f = copy(x30) + _shift

    xliftf = [x1f,x2f,x3f]



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~ BUILD PROBLEMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Discretization
    N = 51
    dt = 0.1

    # Objectives
    q_diag = ones(n_lift)
    # q_diag[4:7] .= 2.
    r_diag = ones(m_lift)
    r_diag[1:4] .= 10.0e-3
    r_diag[5:7] .= 1.0e-6
    Q_lift = [0.1e-2*Diagonal(q_diag), 0.1e-2*Diagonal(q_diag), 0.1e-2*Diagonal(q_diag)]
    Qf_lift = [1000.0*Diagonal(q_diag), 1000.0*Diagonal(q_diag), 1000.0*Diagonal(q_diag)]
    R_lift = Diagonal(r_diag)
    Q_load = 0.0*Diagonal(I,n_load)
    Qf_load = 0.0*Diagonal(I,n_load)
    R_load = 1.0e-6*Diagonal(I,m_load)

    obj_lift = [LQRObjective(Q_lift[i],R_lift,Qf_lift[i],xliftf[i],N) for i = 1:num_lift]
    obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N)

    # Constraints
    constraints_lift = []
    for i = 1:num_lift
        con = Constraints(N)
        for k = 1:N-1
            con[k] += bnd + obs_lift
        end
        con[N] += goal_constraint(xliftf[i])
        push!(constraints_lift,copy(con))
    end

    constraints_load = Constraints(N)
    for k = 1:N-1
        constraints_load[k] += obs_load
    end
    constraints_load[N] += goal_constraint(xloadf)

    # Initial controls
    u_load = [0.;0.;-9.81/num_lift]

    u_lift = zeros(m_lift)
    u_lift[1:4] .= 9.81*(quad_params.m + 1.)/12.
    u_lift[5:7] = u_load
    U0_lift = [u_lift for k = 1:N-1]
    U0_load = [-1.0*[u_load;u_load;u_load] for k = 1:N-1]

    # Create problems
    if agent ∈ 1:num_lift
        i = agent
        prob= Problem(quadrotor_lift,
                        obj_lift[i],
                        U0_lift,
                        integration=:midpoint,
                        constraints=constraints_lift[i],
                        x0=xlift0[i],
                        xf=xliftf[i],
                        N=N,
                        dt=dt)

    elseif agent ∈ [0, :load]
        prob= Problem(doubleintegrator3D_load,
                        obj_load,
                        U0_load,
                        integration=:midpoint,
                        constraints=constraints_load,
                        x0=xload0,
                        xf=xloadf,
                        N=N,
                        dt=dt)
    end
    return prob
end

function build_lift_problem(x0, xf, Q, r_lift, _cyl, num_lift)
    # Discretization
    N = 51
    dt = 0.1

    ### Model
    n_lift = quadrotor_lift.n
    m_lift = quadrotor_lift.m

    ### Constraints
    u_lim_l = -Inf*ones(m_lift)
    u_lim_u = Inf*ones(m_lift)
    u_lim_l[1:4] .= 0.
    u_lim_u[1:4] .= 9.81*(quad_params.m + 1.)/4.0
    bnd = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)

    function cI_cylinder_lift(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 2*r_lift)
        end
    end
    obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

    con = Constraints(N)
    for k = 1:N-1
        con[k] += obs_lift + bnd
    end
    con[N] += goal_constraint(xf)

    ### Objective
    Qf = Diagonal(1000.0I,n_lift)
    r_diag = ones(m_lift)
    r_diag[1:4] .= 10.0e-3
    r_diag[5:7] .= 1.0e-6
    R_lift = Diagonal(r_diag)

    obj_lift = LQRObjective(Q, R_lift, Qf, xf, N)

    u_lift = zeros(m_lift)
    u_lift[1:4] .= 9.81*(quad_params.m + 1.)/12.
    u_lift[5:7] = u_load
    U0_lift = [u_lift for k = 1:N-1]

    prob_lift = Problem(quadrotor_lift,
                obj_lift,
                U0_lift,
                integration=:midpoint,
                constraints=con,
                x0=x0,
                xf=xf,
                N=N,
                dt=dt)
end

function build_load_problem(x0, xf, r_load, _cyl, num_lift)
    # Discretization
    N = 51
    dt = 0.1

    n_load = doubleintegrator3D_load.n
    m_load = doubleintegrator3D_load.m

    # Constraints
    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 2*r_load)
        end
    end
    obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)

    constraints_load = Constraints(N)
    for k = 1:N-1
        constraints_load[k] += obs_load
    end
    constraints_load[N] += goal_constraint(xf)

    # Objective
    Q_load = 0.0*Diagonal(I,n_load)
    Qf_load = 0.0*Diagonal(I,n_load)
    R_load = 1.0e-6*Diagonal(I,m_load)
    obj_load = LQRObjective(Q_load,R_load,Qf_load,xf,N)

    # Initial controls
    u_load = [0.;0.;-9.81/num_lift]
    U0_load = [-1.0*[u_load;u_load;u_load] for k = 1:N-1]

    prob_load = Problem(doubleintegrator3D_load,
                obj_load,
                U0_load,
                integration=:midpoint,
                constraints=constraints_load,
                x0=x0,
                xf=xf,
                N=N,
                dt=dt)
end
