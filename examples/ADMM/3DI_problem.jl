
function build_DI_problem(agent)
    # Set up lift (3x) and load (1x) models
    num_lift = 3
    num_load = 1

    n_slack = 3
    n_lift = Dynamics.doubleintegrator3D.n
    m_lift = Dynamics.doubleintegrator3D.m + n_slack

    #~~~~~~~~~~~~~ DYNAMICS ~~~~~~~~~~~~~~~~#
    function double_integrator_3D_dynamics_lift!(ẋ,x,u) where T
        u_input = u[1:3]
        u_slack = u[4:6]
        Dynamics.double_integrator_3D_dynamics!(ẋ,x,u_input+u_slack)
    end

    doubleintegrator3D_lift = Model(double_integrator_3D_dynamics_lift!,n_lift,m_lift)

    function double_integrator_3D_dynamics_load!(ẋ,x,u) where T
        u_slack1 = u[1:3]
        u_slack2 = u[4:6]
        u_slack3 = u[7:9]
        Dynamics.double_integrator_3D_dynamics!(ẋ,x,u_slack1+u_slack2+u_slack3)
    end

    n_load = Dynamics.doubleintegrator3D.n
    m_load = n_slack*num_lift
    doubleintegrator3D_load = Model(double_integrator_3D_dynamics_load!,n_load,m_load)


    #~~~~~~~~~~~~~ CONSTRAINTS ~~~~~~~~~~~~~~~~#
    # Robot sizes
    r_lift = 0.1
    r_load = 0.1

    # Control limits for lift robots
    u_lim_u = Inf*ones(m_lift)
    u_lim_u[1:3] .= 9.81*2.
    u_lim_l = -Inf*ones(m_lift)
    u_lim_l[3] = 0.
    bnd = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)#,x_min=x_lim_lift_l)

    # Obstacle constraints
    r_cylinder = 0.75

    _cyl = []
    push!(_cyl,(5.,1.,r_cylinder))
    push!(_cyl,(5.,-1.,r_cylinder))

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


    #~~~~~~~~~~~~~ INITIAL CONDITION ~~~~~~~~~~~~~~~~#
    scaling = 1.

    shift_ = zeros(n_lift)
    shift_[1:3] = [0.0;0.0;1.]
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
    N = 21
    dt = 0.1

    # objective
    Q_lift = [1.0e-2*Diagonal(I,n_lift), 10.0e-2*Diagonal(I,n_lift), 0.1e-2*Diagonal(I,n_lift)]
    Qf_lift = [1.0*Diagonal(I,n_lift),1.0*Diagonal(I,n_lift),1.0*Diagonal(I,n_lift)]
    R_lift = 1.0e-4*Diagonal(I,m_lift)

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

function gen_lift_cable_constraints(X_load,U_load,agent,n,m,d,n_slack=3)
    N = length(X_load)
    con_cable_lift = Constraint{Equality}[]
    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                c[1:n_slack] = u[n_slack .+ (1:n_slack)] + U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
            else
                c[1] = norm(x[1:n_slack] - X_load[k][1:n_slack])^2 - d^2
                if k < N
                    c[1 .+ (1:n_slack)] = u[n_slack .+ (1:n_slack)] + U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
                end
            end
        end

        function ∇con(C,x,u=zeros())
            x_pos = x[1:n_slack]
            x_load_pos = X_load[k][1:n_slack]
            dif = x_pos - x_load_pos
            if k == 1
                C[1:n_slack,(n+n_slack) .+ (1:n_slack)] = Is
            else
                C[1,1:n_slack] = 2*dif
                if k < N
                    C[1 .+ (1:n_slack),(n+n_slack) .+ (1:n_slack)] = Is
                end
            end
        end
        if k == 1
            p_con = n_slack
        else
            k < N ? p_con = 1+n_slack : p_con = 1
        end
        # p_con = 1
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_lift)
        push!(con_cable_lift,cc)
    end

    return con_cable_lift
end

function gen_load_cable_constraints(X_lift,U_lift,n,m,d,n_slack=3)
    num_lift = length(X_lift)
    N = length(X_lift[1])
    con_cable_load = Constraint{Equality}[]

    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                _shift = 0
                for i = 1:num_lift
                    c[_shift .+ (1:n_slack)] = U_lift[i][k][n_slack .+ (1:n_slack)] + u[(i-1)*n_slack .+ (1:n_slack)]
                    _shift += n_slack
                end
            else
                for i = 1:num_lift
                    c[i] = norm(X_lift[i][k][1:n_slack] - x[1:n_slack])^2 - d[i]^2
                end

                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        c[_shift .+ (1:n_slack)] = U_lift[i][k][n_slack .+ (1:n_slack)] + u[(i-1)*n_slack .+ (1:n_slack)]
                        _shift += n_slack
                    end
                end
            end
        end

        function ∇con(C,x,u=zeros())
            if k == 1
                _shift = 0
                for i = 1:num_lift
                    u_idx = ((i-1)*n_slack .+ (1:n_slack))
                    C[_shift .+ (1:n_slack),n .+ u_idx] = Is
                    _shift += n_slack
                end
            else
                for i = 1:num_lift
                    x_pos = X_lift[i][k][1:n_slack]
                    x_load_pos = x[1:n_slack]
                    dif = x_pos - x_load_pos
                    C[i,1:n_slack] = -2*dif
                end
                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        u_idx = ((i-1)*n_slack .+ (1:n_slack))
                        C[_shift .+ (1:n_slack),n .+ u_idx] = Is
                        _shift += n_slack
                    end
                end
            end
        end
        if k == 1
            p_con = num_lift*n_slack
        else
            k < N ? p_con = num_lift*(1 + n_slack) : p_con = num_lift
        end
        # p_con = num_lift
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_load)
        push!(con_cable_load,cc)
    end

    return con_cable_load
end

function gen_self_collision_constraints(X_lift,agent,n,m,r_lift,n_slack=3)
    num_lift = length(X_lift)
    N = length(X_lift[1])
    p_con = num_lift - 1

    self_col_con = Constraint{Inequality}[]

    for k = 1:N
        function col_con(c,x,u=zeros())
            p_shift = 1
            for i = 1:num_lift
                if i != agent
                    x_pos = x[1:n_slack]
                    x_pos2 = X_lift[i][k][1:n_slack]
                    c[p_shift] = (r_lift + r_lift)^2 - norm(x_pos - x_pos2)^2
                    p_shift += 1
                end
            end
        end

        function ∇col_con(C,x,u=zeros())
            p_shift = 1
            for i = 1:num_lift
                if i != agent
                    x_pos = x[1:n_slack]
                    x_pos2 = X_lift[i][k][1:n_slack]
                    dif = x_pos - x_pos2
                    C[p_shift,1:n_slack] = -2*dif
                    p_shift += 1
                end
            end
        end

        push!(self_col_con,Constraint{Inequality}(col_con,∇col_con,n,m,p_con,:self_col))
    end

    return self_col_con
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
