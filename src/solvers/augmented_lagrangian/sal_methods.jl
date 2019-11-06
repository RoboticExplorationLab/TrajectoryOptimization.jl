
function solve!(prob::StaticProblem{L,T,StaticALObjective{T}}, solver::StaticALSolver{T,S}) where {L,T,S}
    c_max::T = Inf
    conSet = prob.obj.constraints
    solver.stats.iterations = 0
    solver_uncon = solver.solver_uncon::S
    cost!(prob.obj, Z)
    J_ = get_J(prob.obj)
    J = sum(J_)


    for i = 1:solver.opts.iterations
        set_tolerances!(solver,solver_uncon,i)

        step!(prob, solver)
        J = sum(J_)
        c_max = maximum(conSet.c_max)

        record_iteration!(prob, solver, J, c_max)
        println("Iteration ", i, ": cost = ", J, " c_max = ", c_max)

        converged = evaluate_convergence(solver)
        if converged
            println("Converged at interation $i")
            break
        end

        reset!(solver_uncon)
    end
    return solver
end

function step!(prob::StaticProblem{L,T,<:StaticALObjective}, solver::StaticALSolver{T}) where {L,T}

    # Solve the unconstrained problem
    solve!(prob, solver.solver_uncon)

    # Outer loop update
    dual_update!(prob, solver)
    penalty_update!(prob, solver)
    max_violation!(prob.obj.constraints)
    # copyto!(solver.C_prev,solver.C)

end

function record_iteration!(prob::StaticProblem, solver::StaticALSolver{T,S},
        J::T, c_max::T) where {T,S}

    solver.stats.iterations += 1
    i = solver.stats.iterations
    solver.stats.iterations_total += solver.solver_uncon.stats.iterations
    solver.stats.c_max[i] = c_max
    # solver.stats.
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

function evaluate_convergence(solver::StaticALSolver)
    solver.stats.c_max[solver.stats.iterations] < solver.opts.constraint_tolerance
end

function dual_update!(prob::StaticProblem{L,T,<:StaticALObjective,N,M,NM},
        solver::StaticALSolver) where {L,T,N,M,NM}
    conSet = prob.obj.constraints
    for i in eachindex(conSet.constraints)
        dual_update!(conSet.constraints[i], solver.opts)
    end
end

function penalty_update!(prob::StaticProblem{L,T,<:StaticALObjective,N,M,NM},
        solver::StaticALSolver) where {L,T,N,M,NM}
    conSet = prob.obj.constraints
    for i in eachindex(conSet.constraints)
        penalty_update!(conSet.constraints[i], solver.opts)
    end
end

function max_violation(prob::StaticProblem{L,T,<:StaticALObjective}) where {L,T}
    conSet = prob.obj.constraints
    evaluate(conSet, prob.Z)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end
