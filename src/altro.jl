"ALTRO solve"
function solve!(prob::Problem{T,Discrete},opts::ALTROSolverOptions{T}) where T

    # create ALTRO problem
    prob_altro, state = altro_problem(prob,opts)

    # primary solve (augmented Lagrangian)
    solve!(prob_altro,opts.opts_al)

    # process primary solve results
    process_results!(prob,prob_altro,state,opts)

    return nothing
end

"Processes ALTRO solve results, including: remove slack controls, add minimum time controls"
function process_results!(prob::Problem{T,Discrete},prob_altro::Problem{T,Discrete},
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
                @info "Feasible resolve"
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
function altro_problem(prob::Problem{T,Discrete},opts::ALTROSolverOptions{T}) where T
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
