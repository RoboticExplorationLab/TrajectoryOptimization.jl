
function gen_lift_inequality_constraints(X_load, U_load, n, m)
    N = length(X_load)
    con_height = Constraint{Inequality}[]

    for k = 1:N
        function bnd(c,x,u=zeros())
            c[1] = X_load[k][3] - x[3]
        end
        function ∇bnd(C,x,u=zeros())
            C[1,3] = 1
        end
        ci = Constraint{Inequality}(bnd, n, m, 1, :height)
        push!(con_height, ci)
    end
    return con_height
end

function gen_load_inequality_constraints(X_lift, U_lift, n, m)
    N = length(X_lift[1])
    num_lift = length(X_lift)
    con_height = Constraint{Inequality}[]

    for k = 1:N
        function bnd(c,x,u=zeros())
            for i = 1:num_lift
                c[i] = x[3] - X_lift[i][k][3]
            end
        end
        function ∇bnd(C,x,u=zeros())
            for i = 1:num_lift
                C[i,3] = 1
            end
        end
        ci = Constraint{Inequality}(bnd, n, m, num_lift, :height)
        push!(con_height, ci)
    end
    return con_height
end


function gen_lift_cable_constraints(X_load,U_load,agent,n,m,d,n_slack=3)
    N = length(X_load)
    con_cable_lift = []
    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                c[1:n_slack] = u[(end-(n_slack-1)):end] + U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
            else
                c[1] = norm(x[1:n_slack] - X_load[k][1:n_slack])^2 - d^2
                if k < N
                    c[1 .+ (1:n_slack)] = u[(end-(n_slack-1)):end] + U_load[k][(agent-1)*n_slack .+ (1:n_slack)]
                end
            end
        end

        function ∇con(C,x,u=zeros())
            x_pos = x[1:n_slack]
            x_load_pos = X_load[k][1:n_slack]
            dif = x_pos - x_load_pos
            if k == 1
                C[1:n_slack,(end-(n_slack-1)):end] = Is
            else
                C[1,1:n_slack] = 2*dif
                if k < N
                    C[1 .+ (1:n_slack),(end-(n_slack-1)):end] = Is
                end
            end
        end
        if k == 1
            p_con = n_slack
        else
            k < N ? p_con = 1+n_slack : p_con = 1
        end
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
                    c[_shift .+ (1:n_slack)] = U_lift[i][k][(end-(n_slack-1)):end] + u[(i-1)*n_slack .+ (1:n_slack)]
                    _shift += n_slack
                end
            else
                for i = 1:num_lift
                    c[i] = norm(X_lift[i][k][1:n_slack] - x[1:n_slack])^2 - d[i]^2
                end

                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        c[_shift .+ (1:n_slack)] = U_lift[i][k][(end-(n_slack-1)):end] + u[(i-1)*n_slack .+ (1:n_slack)]
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
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_load)
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
    con_height = gen_lift_inequality_constraints(X_load, U_load, n_lift, m_lift)

    # Add constraints to problems
    for k = 1:N
        prob.constraints[k] += cable_lift[k]
        (k != 1 && k != N) ? prob.constraints[k] += self_col[k] : nothing
        if k > 1
            prob.constraints[k] += con_height[k]
        end
    end
    prob.obj[N].q[3] = -1
end

function update_load_problem(prob, X_lift, U_lift, d::Vector)
    n_load = prob.model.n
    m_load = prob.model.m
    n_slack = 3

    cable_load = gen_load_cable_constraints(X_lift, U_lift, n_load, m_load, d, n_slack)
    con_height = gen_load_inequality_constraints(X_lift, U_lift, n_load, m_load)

    for k = 1:prob.N
        prob.constraints[k] += cable_load[k]
        if k > 1
            prob.constraints[k] += con_height[k]
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
