include("methods.jl")
include("models.jl")

function DI_obstacles()
    r_cylinder = 0.5

    _cyl = []
    push!(_cyl,(5.,1.,r_cylinder))
    push!(_cyl,(5.,-1.,r_cylinder))
    return _cyl
end

function build_DI_problem(agent)
    # Set up lift (3x) and load (1x) models
    num_lift = 3
    num_load = 1

    n_slack = 3
    n_lift = Dynamics.doubleintegrator3D.n
    m_lift = Dynamics.doubleintegrator3D.m + n_slack

    #~~~~~~~~~~~~~ DYNAMICS ~~~~~~~~~~~~~~~~#
    n_load = Dynamics.doubleintegrator3D.n
    m_load = n_slack*num_lift

    #~~~~~~~~~~~~~ CONSTRAINTS ~~~~~~~~~~~~~~~~#
    # Robot sizes
    r_lift = 0.275
    r_load = 0.2

    # Control limits for lift robots
    u_lim_u = Inf*ones(m_lift)
    u_lim_u[1:3] .= 12/.850
    u_lim_l = -Inf*ones(m_lift)
    u_lim_l[3] = 0.
    bnd = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)#,x_min=x_lim_lift_l)

    # Obstacle constraints
    _cyl = DI_obstacles()

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


    #~~~~~~~~~~~~~ INITIAL CONDITION ~~~~~~~~~~~~~~~~#
    scaling = 1.

    shift_ = zeros(n_lift)
    shift_[1:3] = [0.0;0.0;0.5]
    x10 = zeros(n_lift)
    x10[1:3] = scaling*[sqrt(8/9);0.;4/3]
    x10 += shift_
    x20 = zeros(n_lift)
    x20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
    x20 += shift_
    x30 = zeros(n_lift)
    x30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
    x30 += shift_
    xload0 = zeros(n_load)
    xload0[3] = 4/6
    xload0 += shift_

    xlift0 = [x10, x20, x30]

    # goal state
    _shift = zeros(n_lift)
    _shift[1:3] = [10.;0.0;0.0]

    x1f = x10 + _shift
    x2f = x20 + _shift
    x3f = x30 + _shift
    xloadf = xload0 + _shift
    xliftf = [x1f, x2f, x3f]

    d1 = norm(xloadf[1:3]-x1f[1:3])
    d2 = norm(xloadf[1:3]-x2f[1:3])
    d3 = norm(xloadf[1:3]-x3f[1:3])
    d = [d1, d2, d3]



    #~~~~~~~~~~~~~ BUILD PROBLEM ~~~~~~~~~~~~~~~~#

    # discretization
    N = 41
    dt = 0.25

    # objective
    Q_lift = [1.0e-2*Diagonal(I,n_lift), 1.0e-4*Diagonal(I,n_lift), 1.0e-2*Diagonal(I,n_lift)]
    Qf_lift = [1.0*Diagonal(I,n_lift),1.0*Diagonal(I,n_lift),1.0*Diagonal(I,n_lift)]
    R_lift = 1.0*Diagonal(I,m_lift)

    Q_load = 0.0*Diagonal(I,n_load)
    Qf_load = 0.0*Diagonal(I,n_load)
    R_load = 1.0e-4*Diagonal(I,m_load)

    obj_lift = [LQRObjective(Q_lift[i],R_lift,Qf_lift[i],xliftf[i],N) for i = 1:num_lift]
    obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N)

    # constraints
    constraints_lift = Constraints[]
    for i = 1:num_lift
        con = Constraints(N)
        for k = 1:N-1
            con[k] += obs_lift + bnd
        end
        con[N] += goal_constraint(xliftf[i])
        push!(constraints_lift,copy(con))
    end

    constraints_load = Constraints(N)
    for k = 1:N-1
        constraints_load[k] += obs_load #+ bnd3
    end
    constraints_load[N] += goal_constraint(xloadf)


    # Initial Controls
    u_ = [0.;0.;9.81 + 9.81/num_lift;0.;0.;-9.81/num_lift]
    u_load = [0.;0.;9.81/num_lift;0.;0.;9.81/num_lift;0.;0.;9.81/num_lift]

    U0_lift = [u_ for k = 1:N-1]
    U0_load = [u_load for k = 1:N-1]

    # Create Problems
    if agent ∈ 1:num_lift
        prob= Problem(doubleintegrator3D_lift,
                        obj_lift[agent],
                        U0_lift,
                        integration=:midpoint,
                        constraints=constraints_lift[agent],
                        x0=xlift0[agent],
                        xf=xliftf[agent],
                        N=N,
                        dt=dt)
    elseif agent ∈ [0, :load]
        prob = Problem(doubleintegrator3D_load,
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

function update_lift_problem(prob, X_cache, U_cache, agent::Int, d::Float64, r_lift)
    n_lift = prob.model.n
    m_lift = prob.model.m
    n_slack = 3
    N = prob.N

    X_load = X_cache[1]
    U_load = U_cache[1]
    cable_lift = gen_lift_cable_constraints(X_load,
                    U_load,
                    agent,
                    n_lift,
                    m_lift,
                    d,
                    n_slack)


    X_lift = X_cache[2:4]
    U_lift = U_cache[2:4]
    self_col = gen_self_collision_constraints(X_lift, agent, n_lift, m_lift, r_lift, n_slack)

    # Add constraints to problems
    for k = 1:N
        prob.constraints[k] += cable_lift[k]
        (k != 1 && k != N) ? prob.constraints[k] += self_col[k] : nothing
    end
end

function update_load_problem(prob, X_lift, U_lift, d)
    n_load = prob.model.n
    m_load = prob.model.m
    n_slack = 3

    cable_load = gen_load_cable_constraints(X_lift, U_lift, n_load, m_load, d, n_slack)

    for k = 1:prob.N
        prob.constraints[k] += cable_load[k]
    end
end
