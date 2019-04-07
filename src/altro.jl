function solve!(prob::Problem{T},solver::ALTROSolver{T}) where T
    prob_altro = prob
    if !all(x->isnan(x),prob.X[1])
        prob_altro = infeasible_problem(prob_altro,solver.opts.R_inf)
        solver.infeasible
        println("Infeasible Solve")
    end
    if solver.minimum_time
        prob_altro = minimum_time_problem(prob_altro,solver.opts.R_min_time)
        println("Minimum Time Solve")
    end

    solve!(prob_altro,solver.opts.solver_al)
    copyto!(prob.X,prob_altro.X,prob.model.n)
    copyto!(prob.U,prob_altro.U,prob.model.m)
end
