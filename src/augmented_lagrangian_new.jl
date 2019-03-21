function step!(prob::Problem, solver::ALSolver)

    solve!(prob,unconstrained_solver)
    dual_update!(prob,solver)
    penalty_update!(prob,solver)
    results.C_prev .= deepcopy(results.C)

end


function dual_update!(prob::Problem, solver::ALSolver)

end

function penalty_update!(prob::Problem, solver::ALSolver)

end


function λ_update_default!(results::ConstrainedIterResults,solver::Solver)
    n,m,N = get_sizes(solver)
    p,pI,pE = get_num_constraints(solver)
    p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

    for k = 1:N
        k != N ? idx_pI = pI : idx_pI = pI_N

        results.λ[k] = max.(solver.opts.dual_min, min.(solver.opts.dual_max, results.λ[k] + results.μ[k].*results.C[k]))
        results.λ[k][1:idx_pI] = max.(0.0,results.λ[k][1:idx_pI])
    end
end
