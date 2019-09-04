function solve_admm_1slack_dist(probs, prob_load, parallel, opts, n_slack=3)


	N = prob_load.N; dt = prob_load.dt

    # Problem dimensions
    num_lift = length(probs)
    n_lift = 13
    m_lift = 5
    n_load = prob_load.model.n
    m_load = prob_load.model.m


    # Calculate cable lengths based on initial configuration
	x0_load = prob_load.x0
    x0_lift = fetch.([@spawnat w probs[:L].x0 for w in workers()])
    d = [norm(x0_load[1:3]-x0_lift[i][1:3]) for i = 1:num_lift]

	@info "Pre-system solve"
    futures = [@spawnat w solve!(probs[:L], opts_al) for w in workers()]
    solve!(prob_load, opts)
    wait.(futures)

	@info "System solve"

	# Initialize state and control caches
    X_lift = fetch.([@spawnat w deepcopy(probs[:L].X) for w in workers()])
    U_lift = fetch.([@spawnat w deepcopy(probs[:L].U) for w in workers()])
    X_traj = [[prob_load.X]; X_lift]
    U_traj = [[prob_load.U]; U_lift]

    X_cache = ddata(T=Vector{Vector{Vector{Float64}}});
    U_cache = ddata(T=Vector{Vector{Vector{Float64}}});
    @sync for w in workers()
        @spawnat w begin
            X_cache[:L] = X_traj
            U_cache[:L] = U_traj
        end
    end

    X_lift0 = fetch.([@spawnat w probs[:L].X for w in workers()])
    U_lift0 = fetch.([@spawnat w probs[:L].U for w in workers()])
    for i = 1:num_lift
        X_lift[i] .= X_lift0[i]
        U_lift[i] .= U_lift0[i]
    end

    # Update load problem constraints
	update_load!(prob_load,X_lift,U_lift,d)

    # Send trajectories
    @sync for w in workers()
        for i = 2:(num_lift+1)
            @spawnat w begin
                X_cache[:L][i] .= X_lift0[i-1]
                U_cache[:L][i] .= U_lift0[i-1]
            end
        end
        @spawnat w begin
            X_cache[:L][1] .= prob_load.X
            U_cache[:L][1] .= prob_load.U
        end
    end

    # Update lift problems
    @sync for (agent,w) in enumerate(workers())
        @spawnat w update_lift!(probs[:L], agent, X_cache[:L][2:4], X_cache[:L][1], U_cache[:L][1], d[agent])
	end

	solvers_al = ddata(T=AugmentedLagrangianSolver{Float64});
    @sync for w in workers()
        @spawnat w begin
            solvers_al[:L] = AugmentedLagrangianSolver(probs[:L], opts)
            probs[:L] = AugmentedLagrangianProblem(probs[:L],solvers_al[:L])
			probs[:L].model = gen_lift_model(X_cache[:L][1],probs[:L].N,probs[:L].dt)
        end
    end
    solver_load = AugmentedLagrangianSolver(prob_load, opts)
    prob_load = AugmentedLagrangianProblem(prob_load, solver_load)

	prob_load.model = gen_load_model(X_lift,prob_load.N,prob_load.dt)

	for ii = 1:10
        # Solve each AL lift problem
		@info "Solving AL lift problems..."
		if parallel
        	future = [@spawnat w TO.solve_aula!(probs[:L], solvers_al[:L]) for w in workers()]
        	wait.(future)

			# Get Trajectories
			X_lift0 = fetch.([@spawnat w probs[:L].X for w in workers()])
			U_lift0 = fetch.([@spawnat w probs[:L].U for w in workers()])
			for i = 1:num_lift
				X_lift[i] .= X_lift0[i]
				U_lift[i] .= U_lift0[i]
			end
		else
			for (i,w) in enumerate(workers())
				wait(@spawnat w TO.solve_aula!(probs[:L], solvers_al[:L]))
				X_lift[i] .= fetch(@spawnat w probs[:L].X)
				U_lift[i] .= fetch(@spawnat w probs[:L].U)
			end
		end

        # Solve AL load problem
		@info ("Solving load AL problem...")
		prob_load.model = gen_load_model(X_lift,prob_load.N,prob_load.dt)
        TO.solve_aula!(prob_load, solver_load)

        # Send trajectories
		@info "Sending trajectories back..."
        @sync for w in workers()
            for i = 2:(num_lift+1)
                @spawnat w begin
                    X_cache[:L][i] .= X_lift[i-1]
                    U_cache[:L][i] .= U_lift[i-1]
                end
            end
            @spawnat w begin
                X_cache[:L][1] .= prob_load.X
                U_cache[:L][1] .= prob_load.U
				probs[:L].model = gen_lift_model(X_cache[:L][1],probs[:L].N,probs[:L].dt)
            end

        end

        # Update lift constraints prior to evaluating convergence
		@info "Updating constraints"
        @sync for w in workers()
            @spawnat w begin
                TO.update_constraints!(probs[:L].obj.C, probs[:L].obj.constraints, probs[:L].X, probs[:L].U)
                TO.update_active_set!(probs[:L].obj)
            end
        end

        max_c = maximum(fetch.([@spawnat w max_violation(solvers_al[:L]) for w in workers()]))
        max_c = max(max_c, max_violation(solver_load))
		solver_load.stats[:iters_ADMM] = ii
		solver_load.stats[:viol_ADMM] = max_c
        @info max_c
        if max_c < opts.constraint_tolerance
			@info "Solve converged"
            break
		elseif ii == max_iters
			@info "Solve failed to converge"
        end
    end
	@sync for w in workers()
		@spawnat w probs[:L] = update_problem(probs[:L], constraints=probs[:L].obj.constraints)
	end

	solvers = combine_problems(solver_load, solvers_al)
	problems = combine_problems(prob_load, probs)

	return problems, solvers, X_cache
end

function solve_admm_1slack(prob_lift, prob_load, admm_type, opts, n_slack=3)
    N = prob_load.N; dt = prob_load.dt

    # Problem dimensions
    num_lift = length(prob_lift)
    n_lift = prob_lift[1].model.n
    m_lift = prob_lift[1].model.m
    n_load = prob_load.model.n
    m_load = prob_load.model.m

    # Calculate cable lengths based on initial configuration
    d = [norm(prob_lift[i].x0[1:n_slack] - prob_load.x0[1:n_slack]) for i = 1:num_lift]

    for i = 1:num_lift
        solve!(prob_lift[i],opts)

    end
    solve!(prob_load,opts)

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
    solver_lift_al = []
    prob_lift_al = []
    for i = 1:num_lift

        solver = TO.AbstractSolver(prob_lift[i],opts)
        prob = AugmentedLagrangianProblem(prob_lift[i],solver)
        prob.model = gen_lift_model(X_load,N,dt)

        push!(solver_lift_al,solver)
        push!(prob_lift_al,prob)
    end


    solver_load_al = TO.AbstractSolver(prob_load,opts)
    prob_load_al = AugmentedLagrangianProblem(prob_load,solver_load_al)
    prob_load_al.model = gen_load_model(X_lift,N,dt)

    for ii = 1:opts.iterations
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

        prob_load_al.model = gen_load_model(X_lift,N,dt)
        TO.solve_aula!(prob_load_al,solver_load_al)

        # Update constraints
        X_load .= prob_load_al.X
        U_load .= prob_load_al.U

        for i = 1:num_lift
            prob_lift_al[i].model = gen_lift_model(X_load,N,dt)
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

function init_cache(prob_load::Problem, probs::DArray)
    # Initialize state and control caches
    X_lift = fetch.([@spawnat w deepcopy(probs[:L].X) for w in workers()])
    U_lift = fetch.([@spawnat w deepcopy(probs[:L].U) for w in workers()])
    X_traj = [[prob_load.X]; X_lift]
    U_traj = [[prob_load.U]; U_lift]

    X_cache = ddata(T=Vector{Vector{Vector{Float64}}});
    U_cache = ddata(T=Vector{Vector{Vector{Float64}}});
    @sync for w in workers()
        @spawnat w begin
            X_cache[:L] = X_traj
            U_cache[:L] = U_traj
        end
    end
    return X_cache, U_cache, X_lift, U_lift
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

combine_problems(prob_load, probs::Vector) = [[prob_load]; probs]
function combine_problems(prob_load, probs::DArray)
    problems = fetch.([@spawnat w probs[:L] for w in workers()])
    combine_problems(prob_load, problems)
end
