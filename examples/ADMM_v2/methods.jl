
function solve_admm(probs0::Vector{<:Problem}, prob_load0::Problem, quad_params, load_params, parallel, opts; max_iters, n_slack=3)
	probs = copy_probs(probs0)
	prob_load = copy(prob_load0)

	solve_admm_1slack(probs, prob_load, quad_params, load_params, parallel, opts, max_iters, n_slack)
end

copy_probs(probs::Vector{<:Problem}) = copy.(probs)

function solve_admm_1slack(prob_lift, prob_load, quad_params, load_params, admm_type, opts, max_iters=3, n_slack=3)
    N = prob_load.N; dt = prob_load.dt

    # Problem dimensions
    num_lift = length(prob_lift)
    n_lift = prob_lift[1].model.n
    m_lift = prob_lift[1].model.m
    n_load = prob_load.model.n
    m_load = prob_load.model.m

    # Calculate cable lengths based on initial configuration
    d = [norm(prob_lift[i].x0[1:n_slack] - prob_load.x0[1:n_slack]) for i = 1:num_lift]

    #~~~~~~~~~~~~~~~~~~~~ SOLVE INITIAL PROBLEMS ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Solve the initial problems
    for i = 1:num_lift
        solve!(prob_lift[i],opts)
    end
    solve!(prob_load,opts)


    #~~~~~~~~~~~~~~~~~~~~ UPDATE PROBLEMS ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Generate cable constraints
    X_lift = [deepcopy(prob_lift[i].X) for i = 1:num_lift]
    U_lift = [deepcopy(prob_lift[i].U) for i = 1:num_lift]

    X_load = deepcopy(prob_load.X)
    U_load = deepcopy(prob_load.U)

    for i = 1:num_lift
        update_lift!(prob_lift[i],i,X_lift,X_load,U_load,d[i])
    end
    update_load!(prob_load,X_lift,U_lift,d)

    # Create augmented Lagrangian problems, solvers
    solver_lift_al = AugmentedLagrangianSolver{Float64}[]
    prob_lift_al = Problem{Float64,Discrete}[]
    for i = 1:num_lift

        solver = TO.AbstractSolver(prob_lift[i],opts)
        prob = AugmentedLagrangianProblem(prob_lift[i],solver)
        prob.model = gen_lift_model(X_load,N,dt,quad_params)

        push!(solver_lift_al,solver)
        push!(prob_lift_al,prob)
    end

    solver_load_al = TO.AbstractSolver(prob_load,opts)
    prob_load_al = AugmentedLagrangianProblem(prob_load,solver_load_al)
    prob_load_al.model = gen_load_model(X_lift,N,dt,load_params)

    #~~~~~~~~~~~~~~~~~~~~ SOLVE ADMM ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    for ii = 1:max_iters
        # Solve lift agents
        for i = 1:num_lift

            TO.solve_aula!(prob_lift_al[i],solver_lift_al[i])

            # Update constraints (sequentially)
            if admm_type == :sequential
                X_lift[i] .= prob_lift_al[i].X
                U_lift[i] .= prob_lift_al[i].U
            end
        end

        # Update constraints (parallel)
        if admm_type == :parallel
            for i = 1:num_lift
                X_lift[i] .= prob_lift_al[i].X
                U_lift[i] .= prob_lift_al[i].U
            end
        end

        # Solve load
        # return prob_lift,prob_load,1,1
        prob_load_al.model = gen_load_model(X_lift,N,dt,load_params)
        TO.solve_aula!(prob_load_al,solver_load_al)

        # Update constraints
        X_load .= prob_load_al.X
        U_load .= prob_load_al.U

        for i = 1:num_lift
            prob_lift_al[i].model = gen_lift_model(X_load,N,dt,quad_params)
        end

        # Update lift constraints prior to evaluating convergence
        for i = 1:num_lift
            TO.update_constraints!(prob_lift_al[i].obj.C,prob_lift_al[i].obj.constraints, prob_lift_al[i].X, prob_lift_al[i].U)
            TO.update_active_set!(prob_lift_al[i].obj)
        end

        max_c = max([max_violation(solver_lift_al[i]) for i = 1:num_lift]...,max_violation(solver_load_al))
        println(max_c)

        if max_c < opts.constraint_tolerance
            @info "ADMM problem solved"
            break
        end
    end

    return prob_lift_al, prob_load_al, solver_lift_al, solver_load_al
end


function gen_lift_cable_constraints_1slack(X_load,U_load,agent,n,m,d,n_slack=3)
    N = length(X_load)
    con_cable_lift = []
    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                c[1] = u[end] - U_load[k][(agent-1) + 1]
            else
                c[1] = norm(x[1:n_slack] - X_load[k][1:n_slack])^2 - d^2
                if k < N
                    c[2] = u[end] - U_load[k][(agent-1) + 1]
                end
            end
        end

        function ∇con(C,x,u=zeros())
            x_pos = x[1:n_slack]
            x_load_pos = X_load[k][1:n_slack]
            dif = x_pos - x_load_pos
            if k == 1
                C[1,end] = 1.0
            else
                C[1,1:n_slack] = 2*dif
                if k < N
                    C[2,end] = 1.0
                end
            end
        end
        if k == 1
            p_con = 1
        else
            k < N ? p_con = 1+1 : p_con = 1
        end
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_lift)
        push!(con_cable_lift,cc)
    end

    return con_cable_lift
end

function gen_load_cable_constraints_1slack(X_lift,U_lift,n,m,d,n_slack=3)
    num_lift = length(X_lift)
    N = length(X_lift[1])
    con_cable_load = []

    Is = Diagonal(I,n_slack)

    for k = 1:N
        function con(c,x,u=zeros())
            if k == 1
                _shift = 0
                for i = 1:num_lift
                    c[_shift + 1] = U_lift[i][k][end] - u[(i-1) + 1]
                    _shift += 1
                end
            else
                for i = 1:num_lift
                    c[i] = norm(X_lift[i][k][1:n_slack] - x[1:n_slack])^2 - d[i]^2
                end

                if k < N
                    _shift = num_lift
                    for i = 1:num_lift
                        c[_shift + 1] = U_lift[i][k][end] - u[(i-1) + 1]
                        _shift += 1
                    end
                end
            end
        end

        function ∇con(C,x,u=zeros())
            if k == 1
                _shift = 0
                for i = 1:num_lift
                    u_idx = ((i-1) + 1)
                    C[_shift + 1,n + u_idx] = -1.0
                    _shift += 1
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
                        u_idx = ((i-1) + 1)
                        C[_shift + 1,n + u_idx] = -1.0
                        _shift += 1
                    end
                end
            end
        end
        if k == 1
            p_con = num_lift
        else
            k < N ? p_con = num_lift*(1 + 1) : p_con = num_lift
        end
        cc = Constraint{Equality}(con,∇con,n,m,p_con,:cable_load)
        push!(con_cable_load,cc)
    end

    return con_cable_load
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
                    # c[p_shift] = (r_lift + r_lift)^2 - norm(x_pos - x_pos2)^2
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
                    # dif = x_pos - x_pos2
                    # C[p_shift,1:n_slack] = -2*dif
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


function update_lift!(prob_lift,i,X_lift,X_load,U_load,d,n_slack=3)

    n = prob_lift.model.n
    m = prob_lift.model.m
    N = prob_lift.N

    cable_lift = gen_lift_cable_constraints_1slack(X_load,
                    U_load,
                    i,
                    n,
                    m,
                    d,
                    n_slack)


    r_lift = .275
    self_col = gen_self_collision_constraints(X_lift,i,n,m,.275,n_slack)

    # Add system constraints to problems
    for k = 1:N
        prob_lift.constraints[k] += cable_lift[k]
        prob_lift.constraints[k] += self_col[k]
    end
end

function update_load!(prob_load,X_lift,U_lift,d,n_slack=3)
    n = prob_load.model.n
    m = prob_load.model.m
    N = prob_load.N

    cable_load = gen_load_cable_constraints_1slack(X_lift,U_lift,n,m,d,n_slack)

    for k = 1:N
        prob_load.constraints[k] += cable_load[k]
    end
end

function trim_conditions(num_lift,r0_load,quad_params,load_params,quat,opts)
    scenario = :hover
    prob_load_trim = gen_prob(:load, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario)
    prob_lift_trim = [gen_prob(i, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario) for i = 1:num_lift]

    lift_trim, load_trim, slift_al, sload_al = solve_admm(prob_lift_trim, prob_load_trim, quad_params,
            load_params, :sequential, opts_al, max_iters=5)

    scenario = :p2p
    prob_load = gen_prob(:load, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario)

    initial_controls!(prob_load,[load_trim.U[end] for k = 1:prob_load.N])
    # rollout!(prob_load)
    prob_lift = [gen_prob(i, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario) for i = 1:num_lift]

    for i = 1:num_lift
        prob_lift[i].x0[4:7] = lift_trim[i].X[end][4:7]
        prob_lift[i].xf[4:7] = lift_trim[i].X[end][4:7]
        initial_controls!(prob_lift[i],[lift_trim[i].U[end] for k = 1:prob_lift[i].N])
        # rollout!(prob_lift[i])
    end

    return prob_lift, prob_load
end

function trim_conditions_batch(num_lift,r0_load,quad_params,load_params,quat,opts)
    scenario = :hover
    prob_batch_trim = gen_prob(:batch, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario)

	trim, trim_solver = solve(prob_batch_trim,opts)

    scenario = :p2p
    prob_batch = gen_prob(:batch, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario)
	#
    initial_controls!(prob_batch,[trim.U[end] for k = 1:prob_batch.N-1])

    for i = 1:num_lift
        prob_batch.x0[(i-1)*13 .+ (4:7)] = trim.X[end][(i-1)*13 .+ (4:7)]
        prob_batch.xf[(i-1)*13 .+ (4:7)] = trim.X[end][(i-1)*13 .+ (4:7)]
    end

    return prob_batch
end
