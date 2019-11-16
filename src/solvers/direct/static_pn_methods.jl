
function projection_solve!(prob::StaticProblem, solver::StaticPNSolver)
    ϵ_feas = solver.opts.feasibility_tolerance
    viol = norm(solver.d[solver.active_set], Inf)
    max_projection_iters = 10

    count = 0
    while count  max_projection_iters && viol > ϵ_feas
        viol = _projection_solve!(prob, solver)
        count += 1
    end
    return viol
end

function _projection_solve!(prob::StaticProblem, solver::StaticPNSolver)
    Z = primals(solver)
    a = solver.active_set
    max_refinements = 10
    convergence_rate_threshold = 1.1
    ρ = 1e-2

    # Assume constant, diagonal cost Hessian (for now)
    H = Diagonal(solver.H)

    # Update everything
    update_constraints!(prob, solver)
    constraint_jacobian!(prob, solver)
    update_active_set!(prob, solver)
    cost_expansion!(prob, solver)

    # Copy results from constraint sets to sparse arrays
    copy_constraints!(prob, solver)
    copy_jacobians!(prob, solver)

    # Get active constraints
    D,d = active_constraints(prob, solver)

    viol0 = norm(y,Inf)
    if solver.opts.verbose
        println("feas0: $viol0")
    end

    HinvD = H\D'
    S = Symmetric(D*HinvD)
    Sreg = cholesky(S + ρ*I)
    viol_prev = viol0
    count = 0
    while count < max_refinements
        viol = _projection_linesearch!(prob, solver, (S,Sreg), HinvD)
        convergence_rate = log10(viol) / log10(viol_prev)
        viol_prev = viol
        count += 1

        if solver.opts.verbose
            println("conv rate: $convergence_rate")
        end

        if convergence_rate < convergence_rate_threshold ||
                       viol < solver.opts.feasibility_tolerance
            break
        end
    end
    return viol_prev
end

function _projection_linesearch!(prob::StaticProblem, solver::StaticPNSolver,
        S, HinvD)
    a = solver.active_set
    d = solver.d[a]
    viol0 = norm(y,Inf)
    ρ = 1e-4

    Z = primals(solver)
    Z_
    α = 1.0
    ϕ = 0.5
    count = 1
    while true
        δλ = reg_solve(S[1], d, S[2], 1e-8, 25)
        δZ = -HinvD*δλ

        Z_
