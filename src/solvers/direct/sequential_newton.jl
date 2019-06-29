



struct SequentialNewtonSolver{T} <: DirectSolver{T}
    opts::ProjectedNewtonSolverOptions{T}
    stats::Dict{Symbol,Any}
    V::PrimalDual{T}
    Q::Vector{Expansion{T,Diagonal{T,Vector{T}},Diagonal{T,Vector{T}}}}
    Qinv::Vector{Diagonal{T,Vector{T}}}
    Rinv::Vector{Diagonal{T,Vector{T}}}
    fVal::Vector{Vector{T}}
    ∇F::Vector{PartedArray{T,2,Matrix{T},P}} where P
    C::PartedVecTrajectory{T}
    ∇C::Vector{PartedArray{T,2,Matrix{T},P} where P}
    active_set::Vector{Vector{Bool}}
    p::Vector{Int}
end

function SequentialNewtonSolver(prob::Problem{T}, opts::ProjectedNewtonSolverOptions{T}) where T
    n,m,N = size(prob)

    NN = N*n + (N-1)*m
    p = num_constraints(prob)
    pcum = insert!(cumsum(p),1,0)
    P = sum(p) + N*n

    # Partitions
    part_a = (primals=1:NN, duals=NN+1:NN+P) #, ν=NN .+ (1:N*n), λ=NN + N*n .+ (1:sum(p)))
    part_f = create_partition2(prob.model)

    V = PrimalDual(prob)

    # Options
    stats = Dict{Symbol,Any}()

    # Costs
    Q = [Expansion{T,Diagonal,Diagonal}(n,m) for k = 1:N]
    Qinv = [Diagonal(zeros(T,n,n)) for k = 1:N]
    Rinv = [Diagonal(zeros(T,m,m)) for k = 1:N]

    # Constraints
    fVal = [zeros(n) for k = 1:N]
    ∇F = [PartedMatrix(zeros(n,n+m+1),part_f) for k = 1:N]

    constraints = prob.constraints
    C = [PartedVector(con) for con in constraints.C]
    ∇C = [PartedMatrix(con,n,m) for con in constraints.C]
    C[N] = PartedVector(constraints[N],:terminal)
    ∇C[N] = PartedMatrix(constraints[N],n,m,:terminal)

    # Active Set
    active_set = [ones(Bool,pk) for pk in p]

    SequentialNewtonSolver(opts, stats, V, Q, Qinv, Rinv, fVal, ∇F, C, ∇C, active_set, p)
end

function dynamics_jacobian!(prob::Problem, solver::SequentialNewtonSolver, V=solver.V)
    n,m,N = size(prob)
    X,U = V.X, V.U
    solver.∇F[1].xx .= Diagonal(I,n)
    part = (x=1:n, u =n .+ (1:m), x1=n+m .+ (1:n))
    p = solver.p
    for k = 1:N-1
        jacobian!(solver.∇F[k+1], prob.model, X[k], U[k], prob.dt)
    end
end

function active_set!(prob::Problem, solver::SequentialNewtonSolver)
    n,m,N = size(prob)
    for k = 1:N
        active_set!(solver.active_set[k], solver.C[k], solver.opts.active_set_tolerance)
    end
end

function cost_expansion!(prob::Problem, solver::SequentialNewtonSolver, V=solver.V)
    N = prob.N
    X,U = V.X, V.U
    for k = 1:N-1
        cost_expansion!(solver.Q[k], prob.obj[k], X[k], U[k])
        solver.Q[k] / (N-1)
    end
    cost_expansion!(solver.Q[N], prob.obj[N], X[N])
end

function invert_hessian!(prob::Problem, solver::SequentialNewtonSolver)
    N = prob.N
    for k = 1:N-1
        solver.Qinv[k] = inv(solver.Q[k].xx)
        solver.Rinv[k] = inv(solver.Q[k].uu)
    end
    solver.Qinv[N] = inv(solver.Q[N].xx)
end

function update!(prob::Problem, solver::SequentialNewtonSolver)
    dynamics_constraints!(prob, solver)
    dynamics_jacobian!(prob, solver)
    update_constraints!(prob, solver)
    constraint_jacobian!(prob, solver)
    active_set!(prob, solver)
    cost_expansion!(prob, solver)
    invert_hessian!(prob, solver)
    nothing
end

function calc_residual!(prob::Problem, solver::Solver, r)
    N = prob.N

    # Extract variables from solver
    Q = solver.Q
    Qinv = solver.Qinv
    Rinv = solver.Rinv
    a = solver.active_set
    ∇F = solver.∇F
    ∇C = solver.∇C
    c = solver.C

    # Calculate residual
    r[1] = solver.fVal[1] - Qinv[1]*Q[1].x
    for k = 1:N-1
        C = ∇C[k].x[a[k],:]
        D = ∇C[k].u[a[k],:]
        r[2k] = fVal[k] - (∇F[k].xx*Qinv[k]*Q[k].x + ∇F[k].xu*Rinv[k]*Q[k].u - Qinv[k+1]*Q[k+1].x)
        r[2k+1] = c[k][a[k]] - (C*Qinv[k]*Q[k].x + D*Rinv[k]*Q[k].u)
    end
    C = ∇C[N].x[a[N],:]
    r[end] = c[N][a[N]] - C*Qinv[N]*Q[N].x
end

function solve_cholesky(prob::Problem, solver::SequentialNewtonSolver, vals, r)
    n,m,N = size(prob)
    Nb = 2N  # number of blocks
    p_active = sum.(solver.active_set)
    Pa = sum(p_active)

    x,u = SVector(1:n...), SVector(1:m...)

    # Init arrays
    E = vals.E
    F = vals.F
    G = vals.G

    K = vals.K
    L = vals.L
    M = vals.M
    H = vals.H

    r = vals.r
    λ_ = vals.λ_
    λ = vals.λ

    # Extract variables from solver
    a = solver.active_set
    Q = solver.Q
    Qinv = solver.Qinv
    Rinv = solver.Rinv
    fVal = view(solver.fVal, 2:N)
    ∇F = view(solver.∇F, 2:N)
    c = solver.C
    ∇C = solver.∇C

    # Initial condition
    p = p_active[1]
    G0 = cholesky(Qinv[1])
    λ_[1] = G0.L\r[1]

    F[1] = ∇F[1].xx*G0.L
    _G = ∇F[1].xu*Rinv[1]*∇F[1].xu' + Qinv[2]
    G[1] = cholesky(Symmetric(_G))
    λ_[2] = G[1].L\(r[2] - F[1]*λ_[1])

    C = ∇C[1].x[a[1],x]
    D = ∇C[1].u[a[1],u]
    L[1] = C*G0.L
    M[1] = D*Rinv[1]*∇F[1].xu'/(G[1].U)
    H[1] = cholesky(Symmetric(D*Rinv[1]*D' - M[1]*M[1]'))
    λ_[3] = H[1].L\(r[3] - M[1]*λ_[2] - L[1]*λ_[1])

    G_ = G[1]
    M_ = M[1]
    H_ = H[1]
    i = 4
    for k = 2:N-1
        # println("\n Time Step $k")
        p = p_active[k]
        p_ = p_active[k-1]

        # C = (∇C[k].x[a[k],x])
        # D = (∇C[k].u[a[k],u])
        C = (∇C[k].x[solver.active_set[k],:])
        D = (∇C[k].u[solver.active_set[k],:])

        E[k] = -∇F[k].xx*Qinv[k]/G_.U
        F[k] = -E[k]*M_'/H_.U
        G[k] = cholesky(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]' - F[k]*F[k]'))
        λ_[i] = G[k].L\(r[i] - F[k]*λ_[i-1] - E[k]*λ_[i-2])
        i += 1

        L[k] = -K[k]*M_'/H_.U
        M[k] = (C*Qinv[k]*( solver.Q[k].xx - G_.U\(I - (M_'/H_.U) * (H_.L\M_) )/G_.L )*Qinv[k]*∇F[k].xx' + D*Rinv[k]*∇F[k].xu')/G[k].U
        H[k] = cholesky(C*Qinv[k]*C' + D*Rinv[k]*D' - K[k]*K[k]' - L[k]*L[k]' - M[k]*M[k]')
        λ_[i] = H[k].L\(r[i] - M[k]*λ_[i-1] - L[k]*λ_[i-2] - K[k]*λ_[i-3])
        i += 1

        G_, M_, H_ = G[k], M[k], H[k]
    end

    # Terminal
    p = p_active[N]
    p_ = p_active[N-1]

    C = ∇C[N].x[a[N],:]
    D = ∇C[N].u[a[N],:]

    E[N] = -C*Qinv[N]/G_.U
    F[N] = -E[N]*M_'/H_.U
    G[N] = cholesky(Symmetric(C*Qinv[N]*C' - E[N]*E[N]' - F[N]*F[N]'))
    λ_[i] = G[N].L\(r[i] - F[N]*λ_[i-1] - E[N]*λ_[i-2])

    # return λ_

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

    return λ, λ_, r

end

function solve_cholesky_static(prob::Problem, solver::SequentialNewtonSolver, vals)
    n,m,N = size(prob)
    Nb = 2N  # number of blocks
    p_active = sum.(solver.active_set)
    Pa = sum(p_active)

    x,u = SVector(1:n...), SVector(1:m...)

    # Init arrays
    E = vals.E
    F = vals.F
    G = vals.G

    K = vals.K
    L = vals.L
    M = vals.M
    H = vals.H

    r = vals.r
    λ_ = vals.λ_
    λ = vals.λ

    # Extract variables from solver
    a = solver.active_set
    a = vals.a
    Q = solver.Q
    Qinv = solver.Qinv
    Rinv = solver.Rinv
    fVal = view(solver.fVal, 2:N)
    ∇F = view(solver.∇F, 2:N)
    c = solver.C
    ∇C = solver.∇C

    # Calculate residual
    r[1] = solver.fVal[1] - Qinv[1]*Q[1].x
    for k = 1:N-1
        C = ∇C[k].x[a[k],x]
        D = ∇C[k].u[a[k],u]
        r[2k] = fVal[k] - (∇F[k].xx*Qinv[k]*Q[k].x + ∇F[k].xu*Rinv[k]*Q[k].u - Qinv[k+1]*Q[k+1].x)
        r[2k+1] = c[k][a[k]] - (C*Qinv[k]*Q[k].x + D*Rinv[k]*Q[k].u)
    end
    C = ∇C[N].x[a[N],:]
    r[end] = c[N][a[N]] - C*Qinv[N]*Q[N].x


    # Initial condition
    p = p_active[1]
    G0 = cholesky(Qinv[1])
    λ_[1] = G0.L\r[1]

    F[1] = SMatrix{n,n}(∇F[1].xx*G0.L)
    _G = SMatrix{n,n}(∇F[1].xu*Rinv[1]*∇F[1].xu' + Qinv[2])
    G[1] = cholesky(Symmetric(_G))
    λ_[2] = G[1].L\(r[2] - F[1]*λ_[1])

    C = ∇C[1].x[a[1],x]
    D = ∇C[1].u[a[1],u]
    L[1] = SMatrix{p,n}(C*G0.L)
    M[1] = SMatrix{p,n}(D*Rinv[1]*∇F[1].xu'/(G[1].U))
    _H = D*Rinv[1]*D' - M[1]*M[1]'
    H[1] = cholesky(Symmetric(_H))
    λ_[3] = H[1].L\(r[3] - M[1]*λ_[2] - L[1]*λ_[1])

    G_ = G[1]
    M_ = M[1]
    H_ = H[1]
    i = 4
    for k = 2:N-1
        # println("\n Time Step $k")
        p = p_active[k]
        p_ = p_active[k-1]

        C = (∇C[k].x[a[k],x])
        D = (∇C[k].u[a[k],u])
        # C = (∇C[k].x[solver.active_set[k],:])
        # D = (∇C[k].u[solver.active_set[k],:])

        E[k] = -∇F[k].xx*Qinv[k]/G_.U
        if p_ > 0
            F[k] = SMatrix{n,p_}(-E[k]*M_'/H_.U)
            G[k] = cholesky(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]' - F[k]*F[k]'))
            λ_[i] = G[k].L\(r[i] - F[k]*λ_[i-1] - E[k]*λ_[i-2])
        else
            F[k] = @SMatrix zeros(n,0)
            G[k] = cholesky(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]'))
            λ_[i] = G[k].L\(r[i] - E[k]*λ_[i-2])
        end
        i += 1

        if p > 0 #|| true
            K[k] = SMatrix{p,n}(-C*Qinv[k]/G[k].U)
            Q = solver.Q[k].xx
            if p_ > 0
                L[k] = SMatrix{p,p_}(-K[k]*M_'/H_.U)
                _M = (C*Qinv[k]*( Q - G_.U\(I - (M_'/H_.U) * (H_.L\M_) )/G_.L )*Qinv[k]*∇F[k].xx' + D*Rinv[k]*∇F[k].xu')/G[k].U
                M[k] = SMatrix{p,n}(_M)
                H[k] = cholesky(C*Qinv[k]*C' + D*Rinv[k]*D' - K[k]*K[k]' - L[k]*L[k]' - M[k]*M[k]')
                λ_[i] = H[k].L\(r[i] - M[k]*λ_[i-1] - L[k]*λ_[i-2] - K[k]*λ_[i-3])
            else
                L[k] = @SMatrix zeros(p,0)
                _M = (C*Qinv[k]*( Q - G_.U\(I)/G_.L )*Qinv[k]*∇F[k].xx' + D*Rinv[k]*∇F[k].xu')/G[k].U
                M[k] = SMatrix{p,n}(_M)
                H[k] = cholesky(C*Qinv[k]*C' + D*Rinv[k]*D' - K[k]*K[k]' - M[k]*M[k]')
                λ_[i] = H[k].L\(r[i] - M[k]*λ_[i-1] - K[k]*λ_[i-3])
            end
        else
            L[k] = @SMatrix zeros(p_active[k],p_active[k-1])
        end
        i += 1

        G_, M_, H_ = G[k], M[k], H[k]
    end

    # Terminal
    p = p_active[N]
    p_ = p_active[N-1]

    C = ∇C[N].x[a[N],:]
    D = ∇C[N].u[a[N],:]

    E[N] = -C*Qinv[N]/G_.U
    F[N] = SMatrix{p,p_}(-E[N]*M_'/H_.U)
    G[N] = cholesky(Symmetric(C*Qinv[N]*C' - E[N]*E[N]' - F[N]*F[N]'))
    λ_[i] = G[N].L\(r[i] - F[N]*λ_[i-1] - E[N]*λ_[i-2])

    # return λ_

    # BACK SUBSTITUTION
    λ[Nb] = G[N].U\λ_[Nb]
    λ[Nb-1] = H[N-1].U\(λ_[Nb-1] - F[N]'λ[Nb])
    λ[Nb-2] = G[N-1].U\(λ_[Nb-2] - M[N-1]'λ[Nb-1] - E[N]'λ[Nb])

    i = Nb-3
    for k = N-2:-1:1
        # println("\n Time step $k")
        if p_active[k] > 0
            if p_active[k+1] > 0
                λ[i] = H[k].U\(λ_[i] - F[k+1]'λ[i+1] - L[k+1]'λ[i+2])
            else
                λ[i] = H[k].U\(λ_[i] - F[k+1]'λ[i+1])
            end
        end
        i -= 1
        if p_active[k] > 0
            if p_active[k+1] > 0
                λ[i] = G[k].U\(λ_[i] - M[k]'λ[i+1] - E[k+1]'λ[i+2] - K[k+1]'λ[i+3])
            else
                λ[i] = G[k].U\(λ_[i] - M[k]'λ[i+1] - E[k+1]'λ[i+2])
            end
        else
            λ[i] = G[k].U\(λ_[i] - E[k+1]'λ[i+2] - K[k+1]'λ[i+3])
        end
        i -= 1
    end
    λ[1] = G0.U\(λ_[1] - F[1]'λ[2] - L[1]'λ[3])

    return λ, λ_, r

end
