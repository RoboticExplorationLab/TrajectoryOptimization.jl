
function solve_admm(prob_load, probs, opts::TO.AbstractSolverOptions; parallel=true, max_iter=3)
    prob_load = copy(prob_load)
    probs = copy_probs(probs)
    @timeit "init cache" X_cache, U_cache, X_lift, U_lift = init_cache([prob_load; probs])
    @timeit "solve" solvers_al, solver_load = solve_admm!(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts, parallel, max_iter)
	solvers = combine_problems(solver_load, solvers_al)
    problems = combine_problems(prob_load, probs)
	return problems, solvers, X_cache
end

function solve_init!(prob_load, probs::DArray, X_cache, U_cache, X_lift, U_lift, opts)
    # for i in workers()
    #     @spawnat i solve!(probs[:L], opts)
    # end
	num_lift = length(probs)
    futures = [@spawnat w solve!(probs[:L], opts_al) for w in workers()]
    solve!(prob_load, opts)
    wait.(futures)

    # Get trajectories
    X_lift0 = fetch.([@spawnat w probs[:L].X for w in workers()])
    U_lift0 = fetch.([@spawnat w probs[:L].U for w in workers()])
    for i = 1:num_lift
        X_lift[i] .= X_lift0[i]
        U_lift[i] .= U_lift0[i]
    end

    # Update load problem constraints
    x0_load = prob_load.x0
    x0_lift = fetch.([@spawnat w probs[:L].x0 for w in workers()])
    # d1 = norm(x0_load[1:3]-x0_lift[1][1:3])
    # d2 = norm(x0_load[1:3]-x0_lift[2][1:3])
    # d3 = norm(x0_load[1:3]-x0_lift[3][1:3])
    d = [norm(x0_load[1:3]-x0_lift[i][1:3]) for i = 1:num_lift]
    update_load_problem(prob_load, X_lift, U_lift, d)

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
    r_lift = fetch(@spawnat workers()[1] probs[:L].model.info[:radius])::Float64
    @sync for (agent,w) in enumerate(workers())
        @spawnat w update_lift_problem(probs[:L], X_cache[:L], U_cache[:L], agent, d[agent], r_lift, num_lift)
    end
end

function solve_init!(prob_load, probs::Vector{<:Problem}, X_cache, U_cache, X_lift, U_lift, opts)
    num_lift = length(probs)
    for i = 1:num_lift
        solve!(probs[i], opts)
    end
    solve!(prob_load, opts)

    # Get trajectories
    X_lift0 = [prob.X for prob in probs]
    U_lift0 = [prob.U for prob in probs]
    for i = 1:num_lift
        X_lift[i] .= X_lift0[i]
        U_lift[i] .= U_lift0[i]
    end

    # Update Load problem constraints
    x0_load = prob_load.x0
    x0_lift = [prob.x0 for prob in probs]
    # d1 = norm(x0_load[1:3]-x0_lift[1][1:3])
    # d2 = norm(x0_load[1:3]-x0_lift[2][1:3])
    # d3 = norm(x0_load[1:3]-x0_lift[3][1:3])
    d = [norm(x0_load[1:3]-x0_lift[i][1:3]) for i = 1:num_lift]
    update_load_problem(prob_load, X_lift, U_lift, d)

    # Send trajectories
    for w = 2:(num_lift+1)
        for i = 2:(num_lift+1)
            X_cache[w-1][i] .= X_lift0[i-1]
            U_cache[w-1][i] .= U_lift0[i-1]
        end
        X_cache[w-1][1] .= prob_load.X
        U_cache[w-1][1] .= prob_load.U
    end

    # Update lift problems
    r_lift = probs[1].model.info[:radius]::Float64
    for w = 2:(num_lift+1)
        agent = w - 1
        update_lift_problem(probs[agent], prob_load, X_cache[agent], U_cache[agent], agent, num_lift)
    end
end

function solve_admm!(prob_load, probs::Vector{<:Problem}, X_cache, U_cache, X_lift, U_lift, opts, parallel=true, max_iter=3)

    num_lift = length(probs)

    # Solve the initial problems
	println("Solving initial problems...")
    solve_init!(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts)

    # create augmented Lagrangian problems, solvers
	@info "Setting up solvers..."
	if opts isa ALTROSolverOptions
		# for i = 1:num_lift
		# 	probs[i] = TO.altro_problem(probs[i], opts)
		# end
		opts = opts.opts_al
	end
    solvers_al = AugmentedLagrangianSolver{Float64}[]
    for i = 1:num_lift
        solver = AugmentedLagrangianSolver(probs[i],opts)
        probs[i] = AugmentedLagrangianProblem(probs[i],solver)
        push!(solvers_al, solver)
    end
    solver_load = AugmentedLagrangianSolver(prob_load, opts)
    prob_load = AugmentedLagrangianProblem(prob_load, solver_load)

	# @info "Updating constraints..."
    # for i = 1:num_lift
    #     TO.update_constraints!(probs[i].obj.C, probs[i].obj.constraints, probs[i].X, probs[i].U)
    #     TO.update_active_set!(probs[i].obj)
    # end
    # TO.update_constraints!(prob_load.obj.C, prob_load.obj.constraints, prob_load.X, prob_load.U)
    # TO.update_active_set!(prob_load.obj)
	# return solvers_al, solver_load

	max_time = 30.0 # seconds
	t_start = time()
    for ii = 1:max_iter
        # Solve each AL problem
    	@info "Solving AL problems..."
        for i = 1:num_lift
            TO.solve_aula!(probs[i], solvers_al[i])
			if !parallel
				X_lift[i] .= probs[i].X
				U_lift[i] .= probs[i].U
			end
        end

        # Get trajectories
		@info "Solving load problem..."
		if parallel
			for i = 1:num_lift
				X_lift[i] .= probs[i].X
				U_lift[i] .= probs[i].U
			end
		end

        # Solve load with updated lift trajectories
        TO.solve_aula!(prob_load, solver_load)

        # Send trajectories
		@info "Sending trajectories..."
        for i = 1:num_lift  # loop over agents
            for j = 1:num_lift
                i != j || continue
                X_cache[i][j+1] .= X_lift[j]
            end
            X_cache[i][1] .= prob_load.X
        end

        # Update lift constraints prior to evaluating convergence
		@info "Updating constraints..."
        for i = 1:num_lift
            TO.update_constraints!(probs[i].obj.C, probs[i].obj.constraints, probs[i].X, probs[i].U)
            TO.update_active_set!(probs[i].obj)
        end

        max_c = maximum(max_violation.(solvers_al))
        max_c = max(max_c, max_violation(solver_load))
		solver_load.stats[:iters_ADMM] = ii
		solver_load.stats[:viol_ADMM] = max_c
        println(max_c)
        if max_c < opts.constraint_tolerance
            break
        end
		if time() - t_start > max_time
			@warn "Maximum time exceeded"
			break
		end
    end
	for (i,prob) in enumerate(probs)
		probs[i] = update_problem(prob, constraints=prob.obj.constraints)
	end
	return solvers_al, solver_load
end

function solve_admm!(prob_load, probs::DArray, X_cache, U_cache, X_lift, U_lift, opts, parallel=true, max_iters=3)
	num_lift = length(probs)

    @info "Solving initial problems..."
    solve_init!(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts_al)

    # create augmented Lagrangian problems, solvers
    @info "Setting up Solvers..."
    solvers_al = ddata(T=AugmentedLagrangianSolver{Float64});
    @sync for w in workers()
        @spawnat w begin
            solvers_al[:L] = AugmentedLagrangianSolver(probs[:L], opts)
            probs[:L] = AugmentedLagrangianProblem(probs[:L],solvers_al[:L])
        end
    end
    solver_load = AugmentedLagrangianSolver(prob_load, opts)
    prob_load = AugmentedLagrangianProblem(prob_load, solver_load)

	# return solvers_al, solver_load

	max_time = 30.0 # seconds
	t_start = time()
    for ii = 1:max_iters
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
            break
		elseif ii == max_iters
        end
    end
	@sync for w in workers()
		@spawnat w probs[:L] = update_problem(probs[:L], constraints=probs[:L].obj.constraints)
	end
	return solvers_al, solver_load
end

copy_probs(probs::Vector{<:Problem}) = copy.(probs)
function copy_probs(probs::DArray)
    probs2 = ddata(T=eltype(probs))
    @sync for w in workers()
        @spawnat w probs2[:L] = copy(probs[:L])
    end
    return probs2
end


combine_problems(prob_load, probs::Vector) = [[prob_load]; probs]
function combine_problems(prob_load, probs::DArray)
    problems = fetch.([@spawnat w probs[:L] for w in workers()])
    combine_problems(prob_load, problems)
end

function init_cache(probs_all::Vector{<:Problem})
	probs = view(probs_all, 2:4)
	prob_load = probs_all[1]

    num_lift = length(probs)
    X_lift = [deepcopy(prob.X) for prob in probs]
    U_lift = [deepcopy(prob.U) for prob in probs]
    X_traj = [[prob_load.X]; X_lift]
    U_traj = [[prob_load.U]; U_lift]
    X_cache = [deepcopy(X_traj) for i=1:num_lift]
    U_cache = [deepcopy(U_traj) for i=1:num_lift]
    return X_cache, U_cache, X_lift, U_lift
end

function init_cache(probs_all::DArray)
	probs = view(probs_all, 2:4)
	prob_load = probs_all[1]

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