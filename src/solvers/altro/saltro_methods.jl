
function solve!(prob::StaticALProblem, solver::StaticALTROSolver)
    conSet = get_constraints(prob)

    # Set terminal condition if using projected newton
    opts = solver.opts
    if opts.projected_newton
        opts_al = opts.opts_al
        if opts.projected_newton_tolerance >= 0
            opts_al.constraint_tolerance = opts.projected_newton_tolerance
        else
            opts_al.constraint_tolerance = 0
            opts_al.kickout_max_penalty = true
        end
    end

    # Solve with AL
    solve!(prob, solver.solver_al)

    # Check convergence
    i = solver.solver_al.stats.iterations
    c_max = solver.solver_al.stats.c_max[i]

    if c_max > opts.constraint_tolerance
        add_dynamics_constraints!(prob)
        solve!(prob, solver.solver_pn)
    end

end
