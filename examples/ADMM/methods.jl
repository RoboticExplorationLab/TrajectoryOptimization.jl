function gen_lift_cable_constraints(model::Model, X_load,U_load,agent,n_slack=3)
    n,m = model.n, model.m
    N = length(X_load)
    con_cable_lift = []

    for k = 1:N-1
        function con(c,x,u=zeros())

            c[1] = u[end] - U_load[k][(agent-1)+1]

        end

        function ∇con(C,x,u=zeros())
            C[1,end] = 1.0

        end

        cc = Constraint{Equality}(con,∇con,n,m,1,:cable_lift)
        push!(con_cable_lift,cc)
    end

    return con_cable_lift
end

function gen_lift_distance_constraints(model::Model, load_model::Model,
        X_load, U_load, agent, n_slack=3)
    n,m = model.n, model.m
    d = load_model.info[:rope_length]

    N = length(X_load)
    con_cable_lift = []

    for k = 1:N
        function con(c,x,u=zeros())
            c[1] = norm(x[1:n_slack] - X_load[k][1:3])^2 - d^2
        end

        function ∇con(C,x,u=zeros())
            x_pos = x[1:n_slack]
            x_load_pos = X_load[k][1:n_slack]
            dif = x_pos - x_load_pos
            C[1,1:n_slack] = 2*dif
        end
        cc = Constraint{Equality}(con,∇con,n,m,1,:cable_length)
        push!(con_cable_lift,cc)
    end

    return con_cable_lift
end

function gen_load_cable_constraints(model, X_lift, U_lift, n_slack=3)
    n,m = model.n, model.m

    num_lift = length(X_lift)
    N = length(X_lift[1])
    con_cable_load = Constraint{Equality}[]

    for k = 1:N

        function con(c,x,u=zeros())
            _shift = 0
            for i = 1:num_lift
                c[_shift + 1] = U_lift[i][k][end] - u[(i-1)+1]
                _shift += 1
            end
        end

        function ∇con(C,x,u=zeros())
            _shift = 0
            for i = 1:num_lift
                u_idx = ((i-1) + 1)
                C[_shift + 1,n + u_idx] = -1.0
                _shift += 1
            end
        end

        p_con = num_lift
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_load)
        push!(con_cable_load,cc)
    end

    return con_cable_load
end

function gen_load_distance_constraints(model, X_lift, U_lift, n_slack=3)
    n,m = model.n, model.m
    d = model.info[:rope_length]

    num_lift = length(X_lift)
    N = length(X_lift[1])
    con_cable_load = Constraint{Equality}[]


    for k = 1:N

        function con(c,x,u=zeros())
            # r_cables = attachment_points(model, x)
            for i = 1:num_lift
                c[i] = norm(X_lift[i][k][1:n_slack] - x[1:3])^2 - d^2
            end
        end

        function ∇con(C,x,u=zeros())
            for i = 1:num_lift
                x_pos = X_lift[i][k][1:n_slack]
                x_load_pos = x[1:n_slack]
                dif = x_pos - x_load_pos
                C[i,1:n_slack] = -2*dif
            end
        end

        p_con = num_lift
        cc = Constraint{Equality}(con, ∇con, n, m, p_con, :cable_length)
        push!(con_cable_load,cc)
    end

    return con_cable_load
end

function gen_self_collision_constraints(X_lift,agent,n,m,r_lift,n_slack=3)
    num_lift = length(X_lift)
    N = length(X_lift[1])
    p_con = num_lift - 1

    self_col_con = []

    for k = 1:N
        function col_con(c,x,u=zeros())
            p_shift = 1
            for i = 1:num_lift
                if i != agent
                    x_pos = x[1:n_slack]
                    x_pos2 = X_lift[i][k][1:n_slack]
                    c[p_shift] = circle_constraint(x_pos,x_pos2[1],x_pos2[2],2*r_lift)
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

                    C[p_shift,1] = -2*(x_pos[1] - x_pos2[1])
                    C[p_shift,2] = -2*(x_pos[2] - x_pos2[2])
                    p_shift += 1
                end
            end
        end

        push!(self_col_con,Constraint{Inequality}(col_con,∇col_con,n,m,p_con,:self_col))
    end

    return self_col_con
end

function update_lift_problem(prob, prob_load::Problem, X_cache, U_cache, agent::Int, num_lift=3)
    n_lift = prob.model.n
    m_lift = prob.model.m
    n_slack = 3
    r_lift = prob.model.info[:radius]
    d = prob_load.model.info[:rope_length]
    N = prob.N

    X_load = X_cache[1]
    U_load = U_cache[1]
    cable_lift = gen_lift_cable_constraints(prob.model,
                    X_load,
                    U_load,
                    agent,
                    n_slack)
    cable_length = gen_lift_distance_constraints(prob.model, prob_load.model,
                    X_load,
                    U_load,
                    agent,
                    n_slack)

    X_lift = X_cache[2:(num_lift+1)]
    U_lift = U_cache[2:(num_lift+1)]
    self_col = gen_self_collision_constraints(X_lift, agent, n_lift, m_lift, r_lift, n_slack)

    # Add constraints to problems
    for k = 1:N
        if 1 < k < N
            prob.constraints[k] += self_col[k]
            prob.constraints[k] += cable_length[k]
        end

        if k < N
            prob.constraints[k] += cable_lift[k]
        end
    end
end

function update_load_problem(prob, X_lift, U_lift, d::Vector)
    n_load = prob.model.n
    m_load = prob.model.m
    n_slack = 3
    N = prob.N

    cable_load = gen_load_cable_constraints(prob.model, X_lift, U_lift, n_slack)
    cable_length = gen_load_distance_constraints(prob.model, X_lift, U_lift, n_slack)

    for k = 1:N
        if 1 < k < N
            prob.constraints[k] += cable_length[k]
        end
        if k < N
            prob.constraints[k] += cable_load[k]
        end
    end
end


function output_traj(prob,idx=collect(1:6),filename=joinpath(pwd(),"examples/ADMM/traj_output.txt"))
    f = open(filename,"w")
    x0 = prob.x0
    for k = 1:prob.N
        x, y, z, vx, vy, vz = prob.X[k][idx]
        str = "$(x-x0[1]) $(y-x0[2]) $(z) $vx $vy $vz"
        if k != prob.N
            str *= " "
        end
        write(f,str)
    end

    close(f)
end
