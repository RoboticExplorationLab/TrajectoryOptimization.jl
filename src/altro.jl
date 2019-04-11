function solve!(prob::Problem{T},opts::ALTROSolverOptions{T}) where T

    prob_altro = prob

    # create infeasible problem
    if !all(x->isnan(x),prob_altro.X[1])
        println("Infeasible Solve")
        prob_altro = infeasible_problem(prob_altro,opts.R_inf)
    end

    if prob_altro.tf == 0.0
        println("Minimum Time Solve")
        prob_altro = minimum_time_problem(prob_altro,opts.R_min_time)
    end

    # create altro solver
    solver_altro = AbstractSolver(prob_altro,opts)

    # solve
    solve!(prob_altro,solver_altro.solver_al)

    # move results to original problem
    copyto!(prob.X,prob_altro.X,prob.model.n)
    copyto!(prob.U,prob_altro.U,prob.model.m)

    return nothing
end
