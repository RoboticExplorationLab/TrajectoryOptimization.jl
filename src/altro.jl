function solve!(prob::Problem{T},solver::ALTROSolver{T}) where T
    model_altro = prob.model
    con = prob.constraints
    !all(x->isnan(x),prob.X[1]) ? solver.infeasible = true : solver.infeasible = false
    prob_altro = prob
    R_inf = NaN; R_min_time = NaN

    if solver.infeasible
        println("Infeasible Solve")
        R_inf = solver.opts.R_inf
        con_inf = infeasible_constraints(model_altro.n,model_altro.m)
        model_altro = add_slack_controls(model_altro)
        u_slack = slack_controls(prob_altro)
        prob_altro = update_problem(prob_altro,model=model_altro,
            constraints=[prob_altro.constraints...,con_inf],U=[[prob_altro.U[k];u_slack[k]] for k = 1:prob_altro.N-1])
    end

    if solver.minimum_time
        println("Minimum Time Solve")
        R_min_time = solver.opts.R_min_time
        con_min_time_eq, con_min_time_bnd = min_time_constraints(model_altro.n,model_altro.m,1.0,1.0e-3)
        model_altro = add_min_time_controls(model_altro)

        prob_altro = update_problem(prob_altro,model=model_altro,
            constraints=[prob_altro.constraints...,con_min_time_eq,con_min_time_bnd],
            U=[[prob_altro.U[k];prob_altro.dt] for k = 1:prob_altro.N-1],
            X=[[prob_altro.X[k];prob_altro.dt] for k = 1:prob_altro.N],
            x0=[x0;0.0])
    end

    solver_al = AugmentedLagrangianSolver(prob_altro)
    cost_al = AugmentedLagrangianCost(prob_altro,solver_al)
    cost_altro = ALTROCost(prob_altro,cost_al,R_inf,R_min_time)

    prob_altro = update_problem(prob_altro,cost=cost_altro)


    # return prob_altro,solver_al,cost_al,cost_altro
    solver_al = AbstractSolver(prob_altro,solver.opts.solver_al)

    solve!(prob_altro,solver_al)
    copyto!(prob.X,prob_altro.X,prob.model.n)
    copyto!(prob.U,prob_altro.U,prob.model.m)
    return prob_altro, solver_al
end
