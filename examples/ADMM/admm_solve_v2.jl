
function solve_admm(prob_load, probs, opts::TO.AbstractSolverOptions; parallel=true, max_iter=3)
    prob_load = copy(prob_load)
    probs = copy_probs(probs)
    @timeit "init cache" X_cache, U_cache, X_lift, U_lift = init_cache([prob_load; probs])
    @timeit "solve" solvers_al, solver_load = solve_admm!(prob_load, probs, X_cache, U_cache, X_lift, U_lift, opts, parallel, max_iter)
	solvers = combine_problems(solver_load, solvers_al)
    problems = combine_problems(prob_load, probs)
	return problems, solvers, X_cache
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

    d = [norm(x0_load[1:3]-x0_lift[i][1:3]) for i = 1:num_lift]
    # update_load_problem(probs, prob_load, X_lift, U_lift)

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
	N = prob_load.N; dt = prob_load.dt
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
		probs[i].model = gen_lift_model(X_cache[i][1],N,dt)
        push!(solvers_al, solver)
    end
    solver_load = AugmentedLagrangianSolver(prob_load, opts)
    prob_load = AugmentedLagrangianProblem(prob_load, solver_load)
	prob_load.model = gen_load_model(X_cache[1][2:4],N,dt)

	# @info "Updating constraints..."
    # for i = 1:num_lift
    #     TO.update_constraints!(probs[i].obj.C, probs[i].obj.constraints, probs[i].X, probs[i].U)
    #     TO.update_active_set!(probs[i].obj)
    # end
    # TO.update_constraints!(prob_load.obj.C, prob_load.obj.constraints, prob_load.X, prob_load.U)
    # TO.update_active_set!(prob_load.obj)
	# return solvers_al, solver_load

	max_time = 120.0 # seconds
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
		prob_load.model = gen_load_model(X_lift,N,dt)
        TO.solve_aula!(prob_load, solver_load)

		for i = 1:num_lift
			probs[i].model = gen_lift_model(prob_load.X,N,dt)
		end

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


copy_probs(probs::Vector{<:Problem}) = copy.(probs)

combine_problems(prob_load, probs::Vector) = [[prob_load]; probs]


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
