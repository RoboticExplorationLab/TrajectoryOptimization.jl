function solve!(prob::Problem{T},opts::ALTROSolverOptions{T}) where T

    # create ALTRO problem
    prob_altro, state = create_altro_problem(prob,opts)

    # primary solve (augmented Lagrangian)
    solve!(prob_altro,opts.opts_al)

    # # process primary solve results
    process_altro_results!(prob,prob_altro,state,opts)

    return nothing
end

"Processes ALTRO solve results, including: remove slack controls, add minimum time controls"
function process_altro_results!(prob::Problem{T},prob_altro::Problem{T},
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
            prob_altro = infeasible_to_feasible_problem(prob,prob_altro,opts)

            # secondary solve (augmented Lagrangian)
            if opts.resolve_feasible_problem
                solve!(prob_altro,opts.opts_al)
            end
        end

        # update original problem (minimum time solve will include dt controls)
        if state.minimum_time
            for k = 1:prob.N-1
                prob.U[k] = [prob_altro.U[k][1:prob.model.m]; prob_altro.U[k][end]^2]
            end
        end
        copyto!(prob.X,prob_altro.X,prob.model.n)

        # if state.projected_newton
        #     #TODO
        #     nothing
        # end
    end

    return nothing
end

"Return ALTRO problem from original problem, includes: adding slack controls, adding minimum time controls"
function create_altro_problem(prob::Problem{T},opts::ALTROSolverOptions{T}) where T
    prob_altro = prob

    # create infeasible problem
    if !all(x->isnan(x),prob_altro.X[1])
        println("Infeasible Solve")
        prob_altro = infeasible_problem(prob_altro,opts.R_inf)
        infeasible = true
    else
        infeasible = false
    end

    # create minimum time problem
    if prob_altro.tf == 0.0
        println("Minimum Time Solve")
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

"Return a feasible problem from an infeasible problem"
function infeasible_to_feasible_problem(prob::Problem{T},prob_altro::Problem{T},
        state::NamedTuple,opts::ALTROSolverOptions{T}) where T
    prob_altro_feasible = prob

    if state.minimum_time
        prob_altro_feasible = minimum_time_problem(prob_altro_feasible,opts.R_minimum_time,
            opts.dt_max,opts.dt_min)

        # initialize sqrt(dt) from previous solve
        for k = 1:prob.N-1
            prob_altro_feasible.U[k][end] = prob_altro.U[k][end]
            k != 1 ? prob_altro_feasible.X[k][end] = prob_altro.X[k][end] : prob_altro_feasible.X[k][end] = 0.0
        end
        prob_altro_feasible.X[end][end] = prob_altro.X[end][end]
    end

    if opts.dynamically_feasible_projection
        dynamically_feasible_projection!(prob_altro_feasible,opts.opts_al.opts_uncon)
    end

    return prob_altro_feasible
end

"Project dynamically infeasible state trajectory into feasible space using TVLQR"
function dynamically_feasible_projection!(prob::Problem{T},opts::iLQRSolverOptions{T}) where T
    # backward pass - project infeasible trajectory into feasible space using time varying lqr
    solver_ilqr = AbstractSolver(prob,opts)
    backwardpass!(prob, solver_ilqr)

    # rollout
    rollout!(prob,solver_ilqr,0.0)

    # update trajectories
    copyto!(prob.X, solver_ilqr.X̄)
    copyto!(prob.U, solver_ilqr.Ū)
end
