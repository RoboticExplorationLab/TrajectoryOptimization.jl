
function solve!(prob::StaticProblem{L,T,StaticALObjective{T}}, solver::StaticALSolver{T,S}) where {L,T,S}
    solver.stats.iterations = 0
    solver_uncon = solver.solver_uncon::S
    cost!(prob.obj, Z)
    J = sum(prob.obj.obj.J)


    max_violation!(prob.obj.constraints)
    for i = 1:solver.opts.iterations
        set_tolerances!(solver,solver_uncon,i)

        solve!(prob, solver_uncon)
        return
        J = step!(prob_al, solver, solver_uncon)

        record_iteration!(prob, solver, J, solver_uncon)

        converged = evaluate_convergence(solver)
        println(logger,OuterLoop)
        converged ? break : nothing

        reset!(solver_uncon)
    end
    solver.stats[:time] = time() - t_start
    return solver
end

function step!(prob::Problem{T}, solver::AugmentedLagrangianSolver{T},
        unconstrained_solver::AbstractSolver) where T

    # Solve the unconstrained problem
    solve!(prob, unconstrained_solver)
    J = cost(prob)

    # Outer loop update
    dual_update!(prob, solver)
    penalty_update!(prob, solver)
    copyto!(solver.C_prev,solver.C)

    return J
end

function record_iteration!(prob::StaticProblem, solver::StaticALSolver{T}, J::T,
        unconstrained_solver::AbstractSolver) where T

    iter = solver.stats.iterations
    solver.stats.iterations += 1
    solver.stats.iterations_total += unconstrained_solver.stats.iterations
    solver.stats.
    # push!(solver.stats[:iterations_inner], unconstrained_solver.stats[:iterations])
    # push!(solver.stats[:cost],J)
    # push!(solver.stats[:c_max],c_max)
    # push!(solver.stats[:penalty_max],max_penalty(solver))
    # push!(solver.stats_uncon, copy(unconstrained_solver.stats))
    #
end

function set_tolerances!(solver::StaticALSolver{T},
        solver_uncon::AbstractSolver{T},i::Int) where T
    if i != solver.opts.iterations
        solver_uncon.opts.cost_tolerance = solver.opts.cost_tolerance_intermediate
        solver_uncon.opts.gradient_norm_tolerance = solver.opts.gradient_norm_tolerance_intermediate
    else
        solver_uncon.opts.cost_tolerance = solver.opts.cost_tolerance
        solver_uncon.opts.gradient_norm_tolerance = solver.opts.gradient_norm_tolerance
    end

    return nothing
end
