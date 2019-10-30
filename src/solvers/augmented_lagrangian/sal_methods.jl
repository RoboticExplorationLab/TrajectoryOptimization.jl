
function solve!(prob::StaticProblem{T,Discrete}, solver::StaticALSolver{T}) where T<:AbstractFloat
    solver_uncon = solver.solver_uncon

    prob_al = AugmentedLagrangianProblem(prob, solver)
    logger = default_logger(solver)

    rollout!(prob)

    with_logger(logger) do
        record_iteration!(prob_al, solver, cost(prob_al), solver_uncon)
        println(logger,OuterLoop)
        for i = 1:solver.opts.iterations
            set_tolerances!(solver,solver_uncon,i)
            J = step!(prob_al, solver, solver_uncon)

            record_iteration!(prob, solver, J, solver_uncon)

            converged = evaluate_convergence(solver)
            println(logger,OuterLoop)
            converged ? break : nothing

            reset!(solver_uncon)
        end
    end
    solver.stats[:time] = time() - t_start
    return solver
end
