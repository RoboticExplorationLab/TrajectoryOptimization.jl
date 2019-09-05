include("methods.jl")

function solve_admm(probs0::DArray, prob_load0::Problem, parallel, opts, n_slack=3)
	probs = copy_probs(probs0)
	prob_load = copy(prob_load0)
	solve_admm_1slack_dist(probs, prob_load, parallel, opts, n_slack)
end

function copy_probs(probs::DArray)
    probs2 = ddata(T=eltype(probs))
    @sync for w in workers()
        @spawnat w probs2[:L] = copy(probs[:L])
    end
    return probs2
end

function solve_admm_1slack_dist(probs, prob_load, parallel, opts, n_slack=3)


	N = prob_load.N; dt = prob_load.dt

    # Problem dimensions
    num_lift = length(probs)
    # n_lift = 13
    # m_lift = 5
    # n_load = prob_load.model.n
    # m_load = prob_load.model.m

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

	max_iters = 10
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

combine_problems(prob_load, probs::Vector) = [[prob_load]; probs]
function combine_problems(prob_load, probs::DArray)
    problems = fetch.([@spawnat w probs[:L] for w in workers()])
    combine_problems(prob_load, problems)
end
