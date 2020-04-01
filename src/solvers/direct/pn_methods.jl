function solve!(solver::ProjectedNewtonSolver)
    reset!(solver)

    update_constraints!(solver)
    copy_constraints!(solver)
    constraint_jacobian!(solver)
    copy_jacobians!(solver)
    cost_expansion!(solver)
    update_active_set!(solver)
    copy_active_set!(solver)

    if solver.opts.verbose
        println("\nProjection:")
    end
    viol = projection_solve!(solver)
    copyto!(solver.Z, solver.P)
end



function projection_solve!(solver::ProjectedNewtonSolver)
    ϵ_feas = solver.opts.constraint_tolerance
    viol = norm(solver.d[solver.active_set], Inf)
    max_projection_iters = solver.opts.n_steps

    count = 0
    while count < max_projection_iters && viol > ϵ_feas
        viol = _projection_solve!(solver)
        count += 1
        record_iteration!(solver, viol)
    end
    return viol
end

function record_iteration!(solver::ProjectedNewtonSolver, viol)
    solver.stats.iterations += 1
    i = solver.stats.iterations
    solver.stats.cost[i] = cost(solver)
    solver.stats.c_max[i] = viol
end

function reset!(solver::ProjectedNewtonSolver)
    solver.stats.iterations = 0
    solver.stats.cost .*= 0
    solver.stats.c_max .*= 0
    return nothing
end

function _projection_solve!(solver::ProjectedNewtonSolver)
    Z = primals(solver)
    a = solver.active_set
    max_refinements = 10
    convergence_rate_threshold = solver.opts.r_threshold
    ρ = solver.opts.ρ

    # Assume constant, diagonal cost Hessian (for now)
    H = Diagonal(solver.H)

    # Update everything
    update_constraints!(solver)
    constraint_jacobian!(solver)
    update_active_set!(solver)
    cost_expansion!(solver)

    # Copy results from constraint sets to sparse arrays
    copyto!(solver.P, solver.Z)
    copy_constraints!(solver)
    copy_jacobians!(solver)
    copy_active_set!(solver)

    # Get active constraints
    D,d = active_constraints(solver)

    viol0 = norm(d,Inf)
    if solver.opts.verbose
        println("feas0: $viol0")
    end

    HinvD = H\D'
    S = Symmetric(D*HinvD)
    Sreg = cholesky(S + ρ*I)
    viol_prev = viol0
    count = 0
    while count < max_refinements
        viol = _projection_linesearch!(solver, (S,Sreg), HinvD)
        convergence_rate = log10(viol) / log10(viol_prev)
        viol_prev = viol
        count += 1

        if solver.opts.verbose
            println("conv rate: $convergence_rate")
        end

        if convergence_rate < convergence_rate_threshold ||
                       viol < solver.opts.constraint_tolerance
            break
        end
    end
    return viol_prev
end

function _projection_linesearch!(solver::ProjectedNewtonSolver,
        S, HinvD)
    conSet = get_constraints(solver)
    a = solver.active_set
    d = solver.d[a]
    viol0 = norm(d,Inf)
    viol = Inf

    P = solver.P
    Z = solver.Z
    P̄ = solver.P̄
    Z̄ = solver.Z̄

    solve_tol = 1e-8
    refinement_iters = 25
    α = 1.0
    ϕ = 0.5
    count = 1
    while true
        δλ = reg_solve(S[1], d, S[2], solve_tol, refinement_iters)
        δZ = -HinvD*δλ
        P̄.Z .= P.Z + α*δZ

        copyto!(Z̄, P̄)
        update_constraints!(solver, Z̄)
        max_violation!(conSet)
        viol_ = maximum(conSet.c_max)
        copy_constraints!(solver)
        d = solver.d[a]
        viol = norm(d,Inf)

        if solver.opts.verbose
            println("feas: ", viol, " (α = ", α, ")")
        end
        if viol < viol0 || count > 10
            break
        else
            count += 1
            α *= ϕ
        end
    end
    copyto!(P.Z, P̄.Z)
    # copyto!(solver.Z, P.Z)
    return viol
end

reg_solve(A, b, reg::Real, tol=1e-10, max_iters=10) = reg_solve(A, b, A + reg*I, tol, max_iters)
function reg_solve(A, b, B, tol=1e-10, max_iters=10)
    x = B\b
    count = 0
    while count < max_iters
        r = b - A*x
        # println("r_norm = $(norm(r))")

        if norm(r) < tol
            break
        else
            x += B\r
            count += 1
        end
    end
    # println("iters = $count")

    return x
end


function active_constraints(solver::ProjectedNewtonSolver)
    return solver.D[solver.active_set, :], solver.d[solver.active_set]  # this allocates
end

function TrajOptCore.cost_expansion!(solver::ProjectedNewtonSolver)
    Z = get_trajectory(solver)
    E = solver.E
    obj = get_objective(solver)
    cost_expansion!(E, obj, Z)

    xinds, uinds = solver.P.xinds, solver.P.uinds
    H = solver.H
    g = solver.g
    copy_expansion!(H, g, E, xinds, uinds)
    return nothing
end

function copy_expansion!(H, g, E, xinds, uinds)
    N = length(E)

    for k = 1:N-1
        H[xinds[k],xinds[k]] .= E[k].Q
        H[uinds[k],uinds[k]] .= E[k].R
        H[uinds[k],xinds[k]] .= E[k].H
        g[xinds[k]] .= E[k].q
        g[uinds[k]] .= E[k].r
    end
    H[xinds[N],xinds[N]] .= E[N].Q
    g[xinds[N]] .= E[N].q
    return nothing
end

@inline copy_constraints!(solver::ProjectedNewtonSolver) = copy_constraints!(solver.d, solver)
@inline copy_jacobians!(solver::ProjectedNewtonSolver) = copy_jacobians!(solver.D, solver)
@inline copy_active_set!(solver::ProjectedNewtonSolver) = copy_active_set!(solver.active_set, solver)
