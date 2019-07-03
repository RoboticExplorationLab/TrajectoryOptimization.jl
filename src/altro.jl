"ALTRO solve"
function solve!(prob::Problem, opts::ALTROSolverOptions) where T
    t0 = time()

    # Create ALTRO solver
    solver = ALTROSolver(prob, opts)

    # create ALTRO problem
    prob_altro, state = altro_problem(prob,opts)

    # Set terminal condition if using projected newton
    if opts.projected_newton
        opts_al = opts.opts_al
        if opts.projected_newton_tolerance >= 0
            opts_al.constraint_tolerance = opts.projected_newton_tolerance
        else
            opts_al.constraint_tolerance = 0
            opts_al.kickout_max_penalty = true
        end
    end

    # primary solve (augmented Lagrangian)
    t_al = time()
    @info "Augmented Lagrangian solve..."
    solve!(prob_altro, solver.solver_al)
    time_al = time() - t_al

    t_pn = time()
    pn_solver = SequentialNewtonSolver(prob_altro, opts.opts_pn)
    if opts.projected_newton
        @info "Projected Newton solve..."
        solver = solve!(prob_altro, pn_solver)
    end
    time_pn = time()- t_pn

    # process primary solve results
    process_results!(prob,prob_altro,state,opts)

    # Stats
    time_total = time() - t0
    stats = solver.stats
    stats[:time] = time_total
    stats[:time_al] = time_al
    stats[:time_pn] = time_pn

    return solver
end

"Processes ALTRO solve results, including: remove slack controls, add minimum time controls"
function process_results!(prob::Problem{T},prob_altro::Problem{T},
        state::NamedTuple,opts::ALTROSolverOptions{T}) where T

    # move results to original problem
    copyto!(prob.X,prob_altro.X,prob.model.n)
    copyto!(prob.U,prob_altro.U,prob.model.m)

    if !state.infeasible && !state.minimum_time && !state.projected_newton
        return nothing
    else
        # remove infeasible, perform feasible projection, resolve
        if state.infeasible
            # infeasible problem -> feasible problem
            prob_altro = infeasible_to_feasible_problem(prob,prob_altro,state,opts)

            # # secondary solve (augmented Lagrangian)
            if opts.resolve_feasible_problem
                println("Resolving feasible")
                solve!(prob_altro,opts.opts_al)
                copyto!(prob.X,prob_altro.X,prob.model.n)
                copyto!(prob.U,prob_altro.U,prob.model.m)
            end
        end

        # update original problem (minimum time solve will return dt as controls at U[k][end])
        if state.minimum_time
            for k = 1:prob.N-1
                prob.U[k] = [prob_altro.U[k][1:prob.model.m]; prob_altro.U[k][end]^2]
            end
            # copyto!(prob.X,prob_altro.X,prob.model.n)
        end

        # if state.projected_newton
        #     #TODO
        #     nothing
        # end
    end

    return nothing
end

"Return ALTRO problem from original problem, includes: adding slack controls, adding minimum time controls"
function altro_problem(prob::Problem{T},opts::ALTROSolverOptions{T}) where T
    prob_altro = prob

    # create infeasible problem
    if !all(x->isnan(x),prob_altro.X[1])
        @info "Infeasible Solve"
        prob_altro = infeasible_problem(prob_altro,opts.R_inf)
        infeasible = true
    else
        infeasible = false
    end

    # create minimum time problem
    if prob_altro.tf == 0.0
        @info "Minimum Time Solve"
        prob_altro = minimum_time_problem(prob_altro,opts.R_minimum_time,
            opts.dt_max,opts.dt_min)
        minimum_time = true
    else
        minimum_time = false
    end

    state = (infeasible=infeasible,minimum_time=minimum_time,
        projected_newton=opts.projected_newton)

    return prob_altro, state
end
