function active_constraints!(y, solver::SequentialNewtonSolver)
    N = length(solver.Q)
    y[1] = solver.fVal[1]
    a = solver.active_set

    for k = 1:N-1
        y[2k] = solver.fVal[k+1]
        y[2k+1] = solver.C[k][a[k]]
    end
    y[end] = solver.C[N][a[N]]
    nothing
end
function active_constraints(solver::SequentialNewtonSolver)
    y_part = dual_partition(solver)
    y = [zeros(y) for y in y_part]
    active_constraints!(y, solver)
    return y
end

function _projection(solver::SequentialNewtonSolver)
    y = active_constraints(solver)
    calc_factors!(solver)
    δλ = solve_cholesky(solver, y)

    δx,δu = jac_T_mult(solver, δλ)
    δx = -solver.Qinv .* δx
    δu = -solver.Rinv .* δu
    return δx, δu, δλ
end

"""
Solve the least-squares problem to find the best multipliers,
    given the current active multipliers λ
"""
function _mult_projection(solver::SequentialNewtonSolver, λ)
    # Calculate Y'λ
    δx,δu = jac_T_mult(solver, λ)

    # Calculate g + Y'λ
    for k = 1:N-1
        δx[k] += solver.Q[k].x
        δu[k] += solver.Q[k].u
    end
    δx[N] += solver.Q[N].x

    # Calculate Y*(g + Y'λ)
    r = jac_mult(solver, δx, δu)

    # Solve (Y*Y')\r
    eyes = [I for k = 1:N]
    calc_factors!(solver, eyes, eyes)
    δλ = solve_cholesky(solver, r)

    return -δλ
end

function _solveKKT(solver::SequentialNewtonSolver, λ)
    # Calculate r = g + Y'λ
    rx,ru = jac_T_mult(solver, λ)
    for k = 1:N-1
        rx[k] = solver.Q[k].x + rx[k]
        ru[k] = solver.Q[k].u + ru[k]
    end
    rx[N] = solver.Q[N].x + rx[N]

    # Calculate b0 = Y*Hinv*r
    δx = solver.Qinv .* rx
    δu = solver.Rinv .* ru
    b0 = jac_mult(solver, δx, δu)

    # Calculate y - Y*Hinv*r
    y = active_constraints(solver)
    b = y - b0

    # Solve for δλ = (Y*Hinv*Y')\b
    calc_factors!(solver)
    δλ = solve_cholesky(solver, b)

    # Solve for δz = -Hinv*(r + Y'δλ)
    δx, δu = jac_T_mult(solver, δλ)
    for k = 1:N-1
        δx[k] = -solver.Qinv[k]*(δx[k] + rx[k])
        δu[k] = -solver.Rinv[k]*(δu[k] + ru[k])
    end
    δx[N] = -solver.Qinv[N]*(δx[N] + rx[N])
    return δx, δu, δλ
end


######################
#   UPDATE METHODS   #
######################

function _update_primals!(V::PrimalDualVars, δx, δu)
    N = length(V.X)
    for k = 1:N-1
        V.X[k] += δx[k]
        V.U[k] += δu[k]
    end
    V.X[N] += δx[N]
    return nothing
end

function +(V::PrimalDualVars{T}, δz::NTuple{2,Vector{Vector{T}}}) where T
    V_ = copy(V)
    _update_primals!(V_, δz[1], δz[2])
    return V_
end

function _update_duals!(V::PrimalDualVars, δλ, active_set)
    N = length(V.X)
    V.λ[1] += δλ[1]
    for k = 1:N-1
        a = active_set[k]
        V.λ[2k] += δλ[2k]
        λk = view(V.λ[2k+1], a)
        λk .+= δλ[2k+1]
    end
    λN = view(V.λ[2N], active_set[N])
    λN .+= δλ[2N]
end

function +(V::PrimalDualVars, δλ::Tuple{Vector{Vector{T}},Vector{Vector{Bool}}}) where T
    δλ,a = δλ
    V_ = copy(V)
    _update_duals!(V_, δλ, a)
    return V_
end

function +(V::PrimalDualVars,
        δV::Tuple{NTuple{3,Vector{Vector{T}}},Vector{Vector{Bool}}}) where T
    δV,a = δV
    δx,δu,δλ = δV
    V_ = copy(V)
    _update_primals!(V_, δx, δu)
    _update_duals!(V_, δλ, a)
    return V_
end

#######################
#     STATISTICS      #
#######################

function res2(solver::SequentialNewtonSolver, V)
    # Calculate g + Y'λ
    λa = active_duals(V, solver.active_set)
    δx,δu = jac_T_mult(solver, λa)
    for k = 1:N-1
        δx[k] += solver.Q[k].x
        δu[k] += solver.Q[k].u
    end
    δx[N] += solver.Q[N].x
    # return δx, δu, y
    y = active_constraints(solver)
    return sqrt(norm(δx)^2 + norm(δu)^2 + norm(y)^2)
end


##########################
#     SOLVE METHODS      #
##########################

function projection2!(prob, solver::SequentialNewtonSolver, V::PrimalDualVars, active_set_update=true)
    X,U = V.X, V.U
    eps_feasible = solver.opts.feasibility_tolerance
    count = 0
    # cost_expansion!(prob, solver, V)
    feas = Inf
    while true
        dynamics_constraints!(prob, solver, V)
        update_constraints!(prob, solver, V)
        dynamics_jacobian!(prob, solver, V)
        constraint_jacobian!(prob, solver, V)
        if active_set_update
            active_set!(prob, solver)
        end
        y = active_constraints(solver)

        viol = maximum(norm.(y,Inf))
        if solver.opts.verbose
            println("feas: ", viol)
        end
        if viol < eps_feasible || count > 10
            break
        else
            δx, δu = _projection(solver)
            _update_primals!(V, δx, δu)
            count += 1
        end
    end
end

function multiplier_projection!(solver::SequentialNewtonSolver, V)
    λa = active_duals(V, solver.active_set)
    δλ = _mult_projection(solver, λa)
    _update_duals!(V, δλ, solver.active_set)
    return res2(solver,V), δλ
end

function solveKKT(solver::SequentialNewtonSolver, V)
    λa = active_duals(V, solver.active_set)
    δx, δu, δλ = _solveKKT(solver, λa)
    return δx, δu, δλ
end

function newton_step!(prob, solver::SequentialNewtonSolver)
    V = solver.V
    verbose = solver.opts.verbose

    # Initial stats
    update!(prob, solver)
    J0 = cost(prob, V)
    res0 = res2(solver, V)
    viol0 = max_violation(solver)

    # Projection
    verbose ? println("\nProjection:") : nothing
    projection2!(prob, solver, V)
    res1, = multiplier_projection!(solver, V)

    # Solve KKT
    J1 = cost(prob, V)
    viol1 = max_violation(solver)
    δx, δu, δλ = solveKKT(solver, V)

    # Line Search
    verbose ? println("\nLine Search") : nothing
    V_ = line_search(prob, solver, δx, δu, δλ)
    J_ = cost(prob, V_)
    res_ = norm(residual(prob, solver, V_))
    viol_ = max_violation(solver)

    # Print Stats
    if verbose
        println("\nStats")
        println("cost: $J0 → $J1 → $J_")
        println("res: $res0 → $res1 → $res_")
        println("viol: $viol0 → $viol1 → $viol_")
    end

    return V_
end

function line_search(prob::Problem, solver::SequentialNewtonSolver, δx, δu, δλ)
    α = 1.0
    s = 0.01
    J0 = cost(prob, solver.V)
    update!(prob, solver)
    res0 = res2(solver, V)
    count = 0
    solver.opts.verbose ? println("res0: $res0") : nothing
    while count < 10
        V_ = V + (α.*(δx,δu,δλ),solver.active_set)

        # Calculate residual
        projection2!(prob, solver, V_)
        res, = multiplier_projection!(solver, V_)
        J = cost(prob, V_)

        # Calculate max violation
        viol = max_violation(solver)

        if solver.opts.verbose
            println("cost: $J \t residual: $res \t feas: $viol")
        end
        if res < (1-α*s)*res0
            solver.opts.verbose ? println("α: $α") : nothing
            return V_
        end
        count += 1
        α /= 2
    end
    return solver.V
end



"""
Calculate r = Y*z, store result in r, where z is split into x and u
"""
function jac_mult(solver::SequentialNewtonSolver, x, u)
    y_part = dual_partition(solver)
    r = [zeros(y) for y in y_part]

    n,m,N = size(solver)

    # Extract variables from solver
    Q = solver.Q
    Qinv = solver.Qinv
    Rinv = solver.Rinv
    a = solver.active_set
    ∇F = view(solver.∇F,2:N)
    fVal = view(solver.fVal,2:N)
    ∇C = solver.∇C
    c = solver.C

    xi,ui = 1:n, n .+ (1:m)
    C = [∇C[k][a[k],xi] for k = 1:N]
    D = [∇C[k][a[k],ui] for k = 1:N-1]

    # Calculate residual
    r[1] = x[1]
    for k = 1:N-1
        r[2k] = (∇F[k].xx*x[k] + ∇F[k].xu*u[k] - x[k+1])
        r[2k+1] = C[k]*x[k] + D[k]*u[k]
    end
    r[end] = C[N]*x[N]
    return r
end

"""
Calculate x,u = Y'λ
"""
function jac_T_mult(solver::SequentialNewtonSolver, λ)
    n,m,N = size(solver)
    x = [zeros(n) for k = 1:N]
    u = [zeros(m) for k = 1:N-1]

    Q = solver.Q
    ∇F = view(solver.∇F,2:N)
    ∇C = solver.∇C
    a = solver.active_set

    xi,ui = 1:n, n .+ (1:m)
    C = [∇C[k][a[k],xi] for k = 1:N]
    D = [∇C[k][a[k],ui] for k = 1:N-1]

    x[1] = λ[1] + ∇F[1].xx'λ[2] + C[1]'λ[3]
    u[1] =        ∇F[1].xu'λ[2] + D[1]'λ[3]
    for k = 2:N-1
        i = 2k
        x[k] = -λ[i-2] + ∇F[k].xx'λ[i] + C[k]'λ[i+1]
        u[k] =           ∇F[k].xu'λ[i] + D[k]'λ[i+1]
    end
    x[end] = -λ[end-2] + C[N]'λ[end]
    return x,u
end


function calc_factors!(solver::SequentialNewtonSolver, Qinv=solver.Qinv, Rinv=solver.Rinv)
    n,m,N = size(solver)
    Nb = 2N  # number of blocks
    p_active = sum.(solver.active_set)
    Pa = sum(p_active)
    y_parts = dual_partition(solver)

    # Extract arrays
    S_factors = solver.L
    E = S_factors.E
    F = S_factors.F
    G = S_factors.G

    K = S_factors.K
    L = S_factors.L
    M = S_factors.M
    H = S_factors.H


    # Extract variables from solver
    a = solver.active_set
    Q = solver.Q
    fVal = view(solver.fVal, 2:N)
    ∇F = view(solver.∇F, 2:N)
    c = solver.C
    ∇C = solver.∇C

    # Get Active Jacobians
    xi,ui = 1:n, n .+ (1:m)
    C = [∇C[k][a[k],1:n] for k = 1:N]
    D = [∇C[k][a[k],n+1:n+m] for k = 1:N-1]

    # Initial condition
    if Qinv[1] isa UniformScaling
        G0 = cholesky(Array(Diagonal(I,n)))
    else
        G0 = cholesky(Array(Qinv[1]))
    end

    F[1] = ∇F[1].xx*G0.L
    _G = ∇F[1].xu*Rinv[1]*∇F[1].xu' + Qinv[2]
    G[1] = cholesky(Symmetric(_G))

    L[1] = C[1]*G0.L
    M[1] = D[1]*Rinv[1]*∇F[1].xu'/(G[1].U)
    H[1] = cholesky(Symmetric(D[1]*Rinv[1]*D[1]' - M[1]*M[1]'))

    G_ = G[1]
    M_ = M[1]
    H_ = H[1]
    i = 4
    for k = 2:N-1
        E[k] = -∇F[k].xx*Qinv[k]/G_.U
        F[k] = -E[k]*M_'/H_.U
        G[k] = cholesky(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]' - F[k]*F[k]'))
        i += 1

        K[k] = -C[k]*Qinv[k]/G_.U
        L[k] = -K[k]*M_'/H_.U
        M[k] = (C[k]*Qinv[k]*( solver.Q[k].xx - G_.U\(I - (M_'/H_.U) * (H_.L\M_) )/G_.L )*Qinv[k]*∇F[k].xx' + D[k]*Rinv[k]*∇F[k].xu')/G[k].U
        H[k] = cholesky(C[k]*Qinv[k]*C[k]' + D[k]*Rinv[k]*D[k]' - K[k]*K[k]' - L[k]*L[k]' - M[k]*M[k]')
        i += 1

        G_, M_, H_ = G[k], M[k], H[k]
    end

    # Terminal
    E[N] = -C[N]*Qinv[N]/G_.U
    F[N] = -E[N]*M_'/H_.U
    G[N] = cholesky(Symmetric(C[N]*Qinv[N]*C[N]' - E[N]*E[N]' - F[N]*F[N]'))

    # Append G0 onto the end
    push!(G, G0)

    return nothing
end

function solve_cholesky(solver::SequentialNewtonSolver, r)
    n,m,N = size(solver)
    Nb = 2N  # number of blocks
    p_active = sum.(solver.active_set)
    Pa = sum(p_active)
    y_part = dual_partition(solver)


    # Init arrays
    S_factors = solver.L
    E = S_factors.E
    F = S_factors.F
    G = S_factors.G

    K = S_factors.K
    L = S_factors.L
    M = S_factors.M
    H = S_factors.H


    λ = [zeros(y) for y in y_part]
    λ_ = deepcopy(λ)

    # Extract variables from solver
    a = solver.active_set
    Q = solver.Q
    fVal = view(solver.fVal, 2:N)
    ∇F = view(solver.∇F, 2:N)
    c = solver.C
    ∇C = solver.∇C

    # Initial condition
    p = p_active[1]
    G0 = G[end]
    λ_[1] = G0.L\r[1]
    λ_[2] = G[1].L\(r[2] - F[1]*λ_[1])
    λ_[3] = H[1].L\(r[3] - M[1]*λ_[2] - L[1]*λ_[1])

    i = 4
    for k = 2:N-1
        λ_[i] = G[k].L\(r[i] - F[k]*λ_[i-1] - E[k]*λ_[i-2])
        i += 1

        λ_[i] = H[k].L\(r[i] - M[k]*λ_[i-1] - L[k]*λ_[i-2] - K[k]*λ_[i-3])
        i += 1
    end

    # Terminal
    λ_[i] = G[N].L\(r[i] - F[N]*λ_[i-1] - E[N]*λ_[i-2])


    # BACK SUBSTITUTION
    λ[Nb] = G[N].U\λ_[Nb]
    λ[Nb-1] = H[N-1].U\(λ_[Nb-1] - F[N]'λ[Nb])
    λ[Nb-2] = G[N-1].U\(λ_[Nb-2] - M[N-1]'λ[Nb-1] - E[N]'λ[Nb])

    i = Nb-3
    for k = N-2:-1:1
        # println("\n Time step $k")
        λ[i] = H[k].U\(λ_[i] - F[k+1]'λ[i+1] - L[k+1]'λ[i+2])
        i -= 1
        λ[i] = G[k].U\(λ_[i] - M[k]'λ[i+1] - E[k+1]'λ[i+2] - K[k+1]'λ[i+3])
        i -= 1
    end
    λ[1] = G0.U\(λ_[1] - F[1]'λ[2] - L[1]'λ[3])
    return λ
end

function packZ(X,U)
    z = [[x;u] for (x,u) in zip(X[1:end-1],U)]
    push!(z,X[N])
    vcat(z...)
end
