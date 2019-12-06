function solve!(solver::StaticPNSolver)
    update_constraints!(solver)
    copy_constraints!(solver)
    constraint_jacobian!(solver)
    copy_jacobians!(solver)
    cost_expansion!(solver)
    update_active_set!(solver)

    if solver.opts.verbose
        println("\nProjection:")
    end
    viol = projection_solve!(solver)
    copyto!(solver.Z, solver.P)
end



function projection_solve!(solver::StaticPNSolver)
    ϵ_feas = solver.opts.feasibility_tolerance
    viol = norm(solver.d[solver.active_set], Inf)
    max_projection_iters = 10

    count = 0
    while count < max_projection_iters && viol > ϵ_feas
        viol = _projection_solve!(solver)
        count += 1
    end
    return viol
end

function _projection_solve!(solver::StaticPNSolver)
    Z = primals(solver)
    a = solver.active_set
    max_refinements = 10
    convergence_rate_threshold = 1.1
    ρ = 1e-2

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
                       viol < solver.opts.feasibility_tolerance
            break
        end
    end
    return viol_prev
end

function _projection_linesearch!(solver::StaticPNSolver,
        S, HinvD)
    a = solver.active_set
    d = solver.d[a]
    viol0 = norm(d,Inf)
    viol = Inf
    ρ = 1e-4

    P = solver.P
    Z = solver.Z
    P̄ = copy(solver.P)
    Z̄ = solver.Z̄

    α = 1.0
    ϕ = 0.5
    count = 1
    while true
        δλ = reg_solve(S[1], d, S[2], 1e-8, 25)
        δZ = -HinvD*δλ
        P̄.Z .= P.Z + α*δZ

        copyto!(Z̄, P̄)
        update_constraints!(solver, Z̄)
        copy_constraints!(solver)
        d = solver.d[a]
        viol = norm(d,Inf)

        if solver.opts.verbose
            println("feas: ", viol)
        end
        if viol < viol0 || count > 10
            break
        else
            count += 1
            α *= ϕ
        end
    end
    copyto!(P.Z, P̄.Z)
    return viol
end

function update_constraints!(solver::DirectSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    evaluate!(conSet, Z)
end

function update_active_set!(solver::StaticPNSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    update_active_set!(conSet, Z, Val(solver.opts.active_set_tolerance))
    for i = 1:length(conSet.constraints)
        copy_inds(solver.active_set, conSet.constraints[i].active, solver.con_inds[i])
    end
end

function constraint_jacobian!(solver::DirectSolver, Z=get_trajectory(solver))
    conSet = get_constraints(solver)
    jacobian!(conSet, Z)
    return nothing
end

function constraint_jacobian_structure(solver::DirectSolver, Z=get_trajectory(solver))
    n,m,N = size(solver)
    conSet = get_constraints(solver)
    idx = 0.0
    linds = jacobian_linear_inds(solver)

    NN = num_primals(solver)
    NP = num_duals(solver)
    D = spzeros(Int,NP,NN)

    # Number of elements in each block
    blk_len = map(con->length(con.∇c[1]), conSet.constraints)

    # Number of knot points for each constraint
    con_len = map(con->length(con.∇c), conSet.constraints)

    # Linear indices
    for (i,con) in enumerate(conSet.constraints)
        for (j,k) in enumerate(con.inds)
            inds = idx .+ (1:blk_len[i])
            linds[i][j] = inds
            con.∇c[j] = inds
            idx += blk_len[i]
        end
    end
    copy_jacobians!(solver, D)
    return D
end

function active_constraints(solver::StaticPNSolver)
    return solver.D[solver.active_set, :], solver.d[solver.active_set]  # this allocates
end


function cost_expansion!(solver::StaticPNSolver)
    Z = get_trajectory(solver)
    E = solver.E
    obj = get_objective(solver)
    cost_expansion(E, obj, Z)

    xinds, uinds = solver.P.xinds, solver.P.uinds
    H = solver.H
    g = solver.g
    copy_expansion!(H, g, E, xinds, uinds)
    return nothing
end

function copy_expansion!(H, g, E, xinds, uinds)
    N = length(E.x)

    for k = 1:N-1
        H[xinds[k],xinds[k]] .= E.xx[k]
        H[uinds[k],uinds[k]] .= E.uu[k]
        H[uinds[k],xinds[k]] .= E.ux[k]
        g[xinds[k]] .= E.x[k]
        g[uinds[k]] .= E.u[k]
    end
    H[xinds[N],xinds[N]] .= E.xx[N]
    g[xinds[N]] .= E.x[N]
    return nothing
end
