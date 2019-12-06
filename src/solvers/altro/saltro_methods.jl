
function solve!(solver::StaticALTROSolver)
    conSet = get_constraints(solver)

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
    solve!(solver.solver_al)

    # Check convergence
    i = solver.solver_al.stats.iterations
    c_max = solver.solver_al.stats.c_max[i]

    if opts.projected_newton && c_max > opts.constraint_tolerance
        solve!(solver.solver_pn)
    end

end
