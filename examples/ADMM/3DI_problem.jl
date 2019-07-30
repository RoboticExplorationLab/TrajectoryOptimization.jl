include("methods.jl")

function build_lift_problem(x0, xf, Q, r_lift, _cyl, num_lift)
    # discretization
    N = 21
    dt = 0.1

    ### Create model
    n_lift = 6
    m_lift = 6

    function double_integrator_3D_dynamics_lift!(ẋ,x,u) where T
        u_input = u[1:3]
        u_slack = u[4:6]
        Dynamics.double_integrator_3D_dynamics!(ẋ,x,u_input+u_slack)
    end

    doubleintegrator3D_lift = Model(double_integrator_3D_dynamics_lift!,n_lift,m_lift)

    ### Constraints

    # Control limits for lift robots
    u_lim_u = Inf*ones(m_lift)
    u_lim_u[1:3] .= 9.81*2.
    u_lim_l = -Inf*ones(m_lift)
    u_lim_l[3] = 0.

    bnd = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)#,x_min=x_lim_lift_l)

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

    # objective
    Qf = Diagonal(1.0I,n_lift)
    R_lift = 1e-4*Diagonal(1.0I,m_lift)

    obj_lift = LQRObjective(Q, R_lift, Qf, xf, N)

    u_ = [0.;0.;9.81 + 9.81/num_lift; 0.;0.;-9.81/num_lift]
    U0_lift = [u_ for k = 1:N-1]

    prob_lift = Problem(doubleintegrator3D_lift,
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
    # discretization
    N = 21
    dt = 0.1
    n_slack = 3

    function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
        u_slack1 = u[1:3]
        u_slack2 = u[4:6]
        u_slack3 = u[7:9]
        Dynamics.double_integrator_3D_dynamics!(ẋ,x,u_slack1+u_slack2+u_slack3)
    end

    n_load = Dynamics.doubleintegrator3D.n
    m_load = n_slack*num_lift
    doubleintegrator3D_load = Model(double_integrator_3D_dynamics_load!,n_load,m_load)

    # Constraints
    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 2*r_load)
        end
    end
    obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)

    constraints_load = Constraints(N)
    for k = 1:N-1
        constraints_load[k] += obs_load #+ bnd3
    end
    constraints_load[N] += goal_constraint(xf)

    # Objective
    Q_load = 0.0*Diagonal(I,n_load)
    Qf_load = 0.0*Diagonal(I,n_load)
    R_load = 1.0e-4*Diagonal(I,m_load)
    obj_load = LQRObjective(Q_load,R_load,Qf_load,xf,N)

    # Initial controls
    u_load = [0.;0.;9.81/num_lift;0.;0.;9.81/num_lift;0.;0.;9.81/num_lift]
    U0_load = [u_load for k = 1:N-1]

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
