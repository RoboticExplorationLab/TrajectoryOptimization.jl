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
    δλ = solve_cholesky(solver, y)

    δx,δu = jac_T_mult(solver, δλ)
    δx = solver.Qinv .* x
    δu = solver.Rinv .* u
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

    return δλ
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
