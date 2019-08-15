function gen_batch_model(actuated_models,load_model,n_slack=3)
    num_act_models = length(actuated_models)
    nn = zeros(Int,num_act_models)
    mm = zeros(Int,num_act_models)

    for i = 1:num_act_models
        nn[i] = actuated_models[i].n
        mm[i] = actuated_models[i].m
    end

    nn_tol = sum(nn)
    mm_tol = sum(mm)
    n_batch = nn_tol + load_model.n
    m_batch = mm_tol #+ n_slack*num_act_models

    function batch_dynamics!(ẋ,x,u)
        n_shift = 0
        m_shift = 0
        # m_load_shift = copy(mm_tol)

        u_load_tol = zeros(eltype(u),3)

        # update actuated models
        for i = 1:num_act_models
            x_idx = n_shift .+ (1:nn[i])
            u_idx = m_shift .+ (1:mm[i])
            u_load = u[u_idx][end-(n_slack-1):end]

            actuated_models[i].f(view(ẋ,x_idx), x[x_idx], u[u_idx])

            n_shift += nn[i]
            m_shift += mm[i]
            # m_load_shift += n_slack
            u_load_tol += u_load
        end

        # update load
        x_load_idx = (nn_tol .+ (1:load_model.n))
        load_model.f(view(ẋ,x_load_idx),x[x_load_idx],-1.0*u_load_tol/di_mass_load)

        return nothing
    end

    Model(batch_dynamics!,n_batch,m_batch)
end

function gen_batch_load_constraints(actuated_models,load_model,d,n_slack=3)
    num_act_models = length(actuated_models)
    nn = zeros(Int,num_act_models)
    mm = zeros(Int,num_act_models)

    for i = 1:num_act_models
        nn[i] = actuated_models[i].n
        mm[i] = actuated_models[i].m
    end

    nn_tol = sum(nn)
    n_batch = nn_tol + load_model.n
    m_batch = sum(mm) #+ n_slack*num_act_models
    idx_load_pos = (nn_tol .+ (1:load_model.n))[1:n_slack]

    function con(c,x,u=zeros(m_batch))
        n_shift = 0
        x_load_pos = x[idx_load_pos]

        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:n_slack]
            x_pos = x[idx_pos]
            c[i] = norm(x_pos - x_load_pos)^2 - d[i]^2
            n_shift += nn[i]
        end
    end

    function ∇con(C,x,u=zeros(m_batch))
        n_shift = 0
        x_load_pos = x[idx_load_pos]

        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:n_slack]
            x_pos = x[idx_pos]
            dif = x_pos - x_load_pos
            C[i,idx_pos] = 2*dif
            C[i,idx_load_pos] = -2*dif
            n_shift += nn[i]
        end
    end

    Constraint{Equality}(con,∇con,n_batch,m_batch,num_act_models,:load)
end

function gen_batch_self_collision_constraints(actuated_models,load_model,r_act,n_slack=3)

    num_act_models = length(actuated_models)
    nn = zeros(Int,num_act_models)
    mm = zeros(Int,num_act_models)

    for i = 1:num_act_models
        nn[i] = actuated_models[i].n
        mm[i] = actuated_models[i].m
    end

    nn_tol = sum(nn)
    n_batch = nn_tol + load_model.n
    m_batch = sum(mm)# + n_slack*num_act_models

    p_con = 0
    for i = 1:num_act_models
        if i < num_act_models
            for j = (i+1):num_act_models
                p_con += 1
            end
        end
    end

    function col_con(c,x,u=zeros(m_batch))
        n_shift = 0
        p_shift = 1
        for i = 1:num_act_models
            idx_pos = (n_shift .+ (1:nn[i]))[1:n_slack]
            x_pos = x[idx_pos]
            n_shift2 = n_shift + nn[i]
            if i < num_act_models
                for j = (i+1):num_act_models
                    idx_pos2 = (n_shift2 .+ (1:nn[j]))[1:n_slack]
                    x_pos2 = x[idx_pos2]
                    c[p_shift] = circle_constraint(x_pos,x_pos2[1],x_pos2[2],2.5*r_act)
                    n_shift2 += nn[j]
                    p_shift += 1
                end
            end
            n_shift += nn[i]
        end
        @assert p_shift-1 == p_con
    end

    # function ∇col_con(C,x,u=zeros(m_batch))
    #     n_shift = 0
    #     p_shift = 1
    #     for i = 1:num_act_models
    #         idx_pos = (n_shift .+ (1:nn[i]))[1:n_slack]
    #         idx_2dpos = idx_pos[1:end-1]
    #         x_pos = x[idx_pos]
    #
    #         n_shift2 = n_shift + nn[i]
    #         if i < num_act_models
    #             for j = (i+1):num_act_models
    #                 idx_pos2 = (n_shift2 .+ (1:nn[j]))[1:n_slack]
    #                 idx_2dpos2 = idx_pos[1:end-1]
    #                 x_pos2 = x[idx_pos2]
    #                 # dif = x_pos - x_pos2
    #                 # C[p_shift,idx_pos] = -2*dif
    #                 # C[p_shift,idx_pos2] = 2*dif
    #                 jac = [-2*(x_pos[1] - x_pos2[1]),-2*(x_pos[2] - x_pos2[2])]
    #                 C[p_shift,idx_2dpos] = jac
    #                 C[p_shift,idx_2dpos2] = -jac
    #                 n_shift2 += nn[j]
    #                 p_shift += 1
    #             end
    #         end
    #         n_shift += nn[i]
    #     end
    #     @assert p_shift-1 == p_con
    # end

    # Constraint{Inequality}(col_con,∇col_con,n_batch,m_batch,p_con,:col)
    Constraint{Inequality}(col_con,n_batch,m_batch,p_con,:col)

end
