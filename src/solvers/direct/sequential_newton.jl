
##########################################
#            CUSTOM TYPES                #
##########################################

struct PrimalDualVars{T}
    X::VectorTrajectory{T}
    U::VectorTrajectory{T}
    λ::Matrix{Vector{T}}
end

function PrimalDualVars(prob::Problem)
    n,m,N = size(prob)
    X = prob.X
    U = prob.U

    p = num_constraints(prob)
    y_part = dual_partition(n,m,p,N)
    λ = [ zeros(y) for y in y_part, i=1:1]
    PrimalDualVars(X,U,λ)
end

function copy(V::PrimalDualVars)
    PrimalDualVars(deepcopy(V.X), deepcopy(V.U), deepcopy(V.λ))
end

struct KKTFactors{T}
    E::MatrixTrajectory{T}
    F::MatrixTrajectory{T}
    G::Vector{Cholesky{T,Matrix{T}}}
    K::MatrixTrajectory{T}
    L::MatrixTrajectory{T}
    M::MatrixTrajectory{T}
    H::Vector{Cholesky{T,Matrix{T}}}
end

function KKTFactors(n,m,p_active,N)
    E = [zeros(n,n) for p in p_active]
    F = [zeros(n,p) for p in p_active]
    G = [cholesky(Matrix(I,n,n)) for p in p_active]

    K = [zeros(p,n) for p in p_active]
    L = [zeros(p,p) for p in p_active]
    M = [zeros(p,n) for p in p_active]
    H = [cholesky(Matrix(I,p,p)) for p in p_active]
    KKTFactors(E,F,G,K,L,M,H)
end



struct SequentialNewtonSolver{T} <: DirectSolver{T}
    opts::ProjectedNewtonSolverOptions{T}
    stats::Dict{Symbol,Any}
    V::PrimalDualVars{T}
    V_::PrimalDualVars{T}
    δx::VectorTrajectory{T}
    δu::VectorTrajectory{T}
    δλ::Matrix{Vector{T}}
    r::Matrix{Vector{T}}
    L::KKTFactors{T}

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
    y_part = dual_partition(n,m,p,N)

    # Options
    stats = Dict{Symbol,Any}()

    # PrimalDuals Variables
    V = PrimalDualVars(copy(prob))
    V_ = copy(V)

    # Deviations
    δx = [zeros(n) for k = 1:N]
    δu = [zeros(m) for k = 1:N-1]
    δλ = [zeros(y) for y in y_part, i=1:1]
    r  = [zeros(y) for y in y_part, i=1:1]

    # KKT Factors
    L = KKTFactors(n,m,p,N)

    # Costs
    Q = [Expansion{T,Diagonal,Diagonal}(n,m) for k = 1:N]
    Qinv = [Diagonal(zeros(T,n,n)) for k = 1:N]
    Rinv = [Diagonal(zeros(T,m,m)) for k = 1:N-1]

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

    SequentialNewtonSolver(opts, stats, V, copy(V), δx, δu, δλ, r, L, Q, Qinv, Rinv, fVal, ∇F, C, ∇C, active_set, p)
end

##########################################
#        TYPE UTIL FUNCTIONS             #
##########################################

function size(solver::SequentialNewtonSolver)
    n,m = length(solver.δx[1]), length(solver.δu[1])
    N = length(solver.Q)
    return n,m,N
end

function num_active_constraints(solver::SequentialNewtonSolver)
    n,m,N = size(solver)
    sum(sum.(solver.active_set)) + N*n
end

"""
Construct 2N partition of dual variables (n,n,p1,n,p2,n,p3,...,n,pN-1,pN)
"""
function dual_partition(n,m,p,N)
    y_part = ones(Int,2,N-1)*n
    y_part[2,:] = p[1:end-1]
    y_part = vec(y_part)
    insert!(y_part,1,3)
    push!(y_part, p[N])
    return y_part
end

function dual_partition(solver::SequentialNewtonSolver)
    n,m,N = size(solver)
    p_active = sum.(solver.active_set)
    dual_partition(n,m,p_active,N)
end

function update!(prob::Problem, solver::SequentialNewtonSolver, V=solver.V, active_set=true)
    dynamics_constraints!(prob, solver, V)
    dynamics_jacobian!(prob, solver, V)
    update_constraints!(prob, solver, V)
    constraint_jacobian!(prob, solver, V)
    if active_set
        active_set!(prob, solver)
    end
    cost_expansion!(prob, solver, V)
    invert_hessian!(prob, solver)
    nothing
end

function active_duals(V::PrimalDualVars{T}, a)::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}} where T
    N = length(V.X)
    n = length(V.X[1])
    dyn = collect(1:n)
    λ = [begin
            if i == 1 || iseven(i)
                view(V.λ[i],dyn)
            elseif isodd(i) || i == 2N
                k = (i-1)÷2
                view(V.λ[i],a[k])
            end
        end for i = 1:2N
        ]
    return λ
end

"""
Take the 2-norm of the residual: [g + Y'λ; y]
"""
function res2(solver::SequentialNewtonSolver, V)
    N = length(solver.Q)

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

function packZ(X,U)
    N = length(X)
    z = [[x;u] for (x,u) in zip(X[1:end-1],U)]
    push!(z,X[N])
    vcat(z...)
end


#########################################
#         CONSTRAINT FUNCTIONS          #
#########################################

function dynamics_jacobian!(prob::Problem, ∇F, X, U)
    n,m,N = size(prob)
    ∇F[1].xx .= Diagonal(I,n)
    dt = get_dt_traj(prob,U)
    for k = 1:N-1
        jacobian!(∇F[k+1], prob.model, X[k], U[k], dt[k])
    end
end
dynamics_jacobian!(prob::Problem, solver::SequentialNewtonSolver, V=solver.V) =
    dynamics_jacobian!(prob, solver.∇F, V.X, V.U)


function active_set!(prob::Problem, solver::SequentialNewtonSolver)
    n,m,N = size(prob)
    for k = 1:N
        active_set!(solver.active_set[k], solver.C[k], solver.opts.active_set_tolerance)
    end
end

"""
Get the 2N vector of active constraints
"""
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


#########################################
#       COST EXPANSION FUNCTIONS        #
#########################################

function cost_expansion!(prob::Problem, solver::SequentialNewtonSolver, V=solver.V)
    N = prob.N
    X,U, dt = V.X, V.U, get_dt_traj(prob,V.U)
    for k = 1:N-1
        cost_expansion!(solver.Q[k], prob.obj[k], X[k], U[k], dt[k])
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



#########################################
#     CORE LINEAR ALGEBRA FUNCTIONS     #
#########################################

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

"""
Compute the cholesky factorization of (Y*Hinv*Y')
where Y is the constraint jacobian and Hinv is the inverse of the cost Hessian
"""
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
        G0 = cholesky_reg(Array(Diagonal(I,n)))
    else
        G0 = cholesky_reg(Array(Qinv[1]))
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
        G[k] = cholesky_reg(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]' - F[k]*F[k]'))
        G[k].info != 0 ? println("failed G cholesky at k = $k") : nothing
        i += 1

        K[k] = -C[k]*Qinv[k]/G_.U
        L[k] = -K[k]*M_'/H_.U
        M[k] = (C[k]*Qinv[k]*( solver.Q[k].xx - G_.U\(I - (M_'/H_.U) * (H_.L\M_) )/G_.L )*Qinv[k]*∇F[k].xx' + D[k]*Rinv[k]*∇F[k].xu')/G[k].U
        H[k] = cholesky_reg(C[k]*Qinv[k]*C[k]' + D[k]*Rinv[k]*D[k]' - K[k]*K[k]' - L[k]*L[k]' - M[k]*M[k]')
        H[k].info != 0 ? println("failed H cholesky at k = $k") : nothing
        i += 1

        G_, M_, H_ = G[k], M[k], H[k]
    end

    # Terminal
    E[N] = -C[N]*Qinv[N]/G_.U
    F[N] = -E[N]*M_'/H_.U
    G[N] = cholesky_reg(Symmetric(C[N]*Qinv[N]*C[N]' - E[N]*E[N]' - F[N]*F[N]'))
    G[N].info != 0 ? println("failed G cholesky at k = N") : nothing

    # Append G0 onto the end
    push!(G, G0)

    return nothing
end

function cholesky_reg(A::AbstractMatrix)
    C = cholesky(A, check=true)
    if C.info != 0 && false
        E = eigen(A)
        v = min(minimum(E.values),-1e-2)
        cholesky(A - 2I*v)
    end
    return C
end


"""
Solve the system S*δλ = r
    where S = (Y*Hinv*Y') and is already factored and stored in a `KKTFactors` type
"""
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
# function solve_cholesky(prob::Problem, solver::SequentialNewtonSolver, r,
#         Qinv=solver.Qinv, Rinv=solver.Rinv)
#     n,m,N = size(prob)
#     Nb = 2N  # number of blocks
#     p_active = sum.(solver.active_set)
#     Pa = sum(p_active)
#
#     x,u = SVector(1:n...), SVector(1:m...)
#
#     # Init arrays
#     S_factors = solver.L
#     E = S_factors.E
#     F = S_factors.F
#     G = S_factors.G
#
#     K = S_factors.K
#     L = S_factors.L
#     M = S_factors.M
#     H = S_factors.H
#
#     r = solver.r
#     λ = solver.δλ
#     λ_ = deepcopy(λ)
#
#     # Extract variables from solver
#     a = solver.active_set
#     Q = solver.Q
#     fVal = view(solver.fVal, 2:N)
#     ∇F = view(solver.∇F, 2:N)
#     c = solver.C
#     ∇C = solver.∇C
#
#     # Initial condition
#     p = p_active[1]
#     if Qinv[1] isa UniformScaling
#         G0 = cholesky(Diagonal(I,n))
#     else
#         G0 = cholesky(Qinv[1])
#     end
#     λ_[1] = G0.L\r[1]
#
#     F[1] = ∇F[1].xx*G0.L
#     _G = ∇F[1].xu*Rinv[1]*∇F[1].xu' + Qinv[2]
#     G[1] = cholesky(Symmetric(_G))
#     λ_[2] = G[1].L\(r[2] - F[1]*λ_[1])
#
#     C = ∇C[1].x[a[1],x]
#     D = ∇C[1].u[a[1],u]
#     L[1] = C*G0.L
#     M[1] = D*Rinv[1]*∇F[1].xu'/(G[1].U)
#     H[1] = cholesky(Symmetric(D*Rinv[1]*D' - M[1]*M[1]'))
#     λ_[3] = H[1].L\(r[3] - M[1]*λ_[2] - L[1]*λ_[1])
#
#     G_ = G[1]
#     M_ = M[1]
#     H_ = H[1]
#     i = 4
#     for k = 2:N-1
#         # println("\n Time Step $k")
#         p = p_active[k]
#         p_ = p_active[k-1]
#
#         C = (∇C[k].x[solver.active_set[k],:])
#         D = (∇C[k].u[solver.active_set[k],:])
#
#         E[k] = -∇F[k].xx*Qinv[k]/G_.U
#         F[k] = -E[k]*M_'/H_.U
#         G[k] = cholesky(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]' - F[k]*F[k]'))
#         λ_[i] = G[k].L\(r[i] - F[k]*λ_[i-1] - E[k]*λ_[i-2])
#         i += 1
#
#         K[k] = -C*Qinv[k]/G_.U
#         L[k] = -K[k]*M_'/H_.U
#         M[k] = (C*Qinv[k]*( solver.Q[k].xx - G_.U\(I - (M_'/H_.U) * (H_.L\M_) )/G_.L )*Qinv[k]*∇F[k].xx' + D*Rinv[k]*∇F[k].xu')/G[k].U
#         H[k] = cholesky(C*Qinv[k]*C' + D*Rinv[k]*D' - K[k]*K[k]' - L[k]*L[k]' - M[k]*M[k]')
#         λ_[i] = H[k].L\(r[i] - M[k]*λ_[i-1] - L[k]*λ_[i-2] - K[k]*λ_[i-3])
#         i += 1
#
#         G_, M_, H_ = G[k], M[k], H[k]
#     end
#
#     # Terminal
#     p = p_active[N]
#     p_ = p_active[N-1]
#
#     C = ∇C[N].x[a[N],:]
#     D = ∇C[N].u[a[N],:]
#
#     E[N] = -C*Qinv[N]/G_.U
#     F[N] = -E[N]*M_'/H_.U
#     G[N] = cholesky(Symmetric(C*Qinv[N]*C' - E[N]*E[N]' - F[N]*F[N]'))
#     λ_[i] = G[N].L\(r[i] - F[N]*λ_[i-1] - E[N]*λ_[i-2])
#
#     # return λ_
#
#     # BACK SUBSTITUTION
#     λ[Nb] = G[N].U\λ_[Nb]
#     λ[Nb-1] = H[N-1].U\(λ_[Nb-1] - F[N]'λ[Nb])
#     λ[Nb-2] = G[N-1].U\(λ_[Nb-2] - M[N-1]'λ[Nb-1] - E[N]'λ[Nb])
#
#     i = Nb-3
#     for k = N-2:-1:1
#         # println("\n Time step $k")
#         λ[i] = H[k].U\(λ_[i] - F[k+1]'λ[i+1] - L[k+1]'λ[i+2])
#         i -= 1
#         λ[i] = G[k].U\(λ_[i] - M[k]'λ[i+1] - E[k+1]'λ[i+2] - K[k+1]'λ[i+3])
#         i -= 1
#     end
#     λ[1] = G0.U\(λ_[1] - F[1]'λ[2] - L[1]'λ[3])
#
#     return λ, λ_, r
#
# end
#
# function solve_cholesky_static(prob::Problem, solver::SequentialNewtonSolver, vals)
#     n,m,N = size(prob)
#     Nb = 2N  # number of blocks
#     p_active = sum.(solver.active_set)
#     Pa = sum(p_active)
#
#     x,u = SVector(1:n...), SVector(1:m...)
#
#     # Init arrays
#     E = vals.E
#     F = vals.F
#     G = vals.G
#
#     K = vals.K
#     L = vals.L
#     M = vals.M
#     H = vals.H
#
#     r = vals.r
#     λ_ = vals.λ_
#     λ = vals.λ
#
#     # Extract variables from solver
#     a = solver.active_set
#     a = vals.a
#     Q = solver.Q
#     Qinv = solver.Qinv
#     Rinv = solver.Rinv
#     fVal = view(solver.fVal, 2:N)
#     ∇F = view(solver.∇F, 2:N)
#     c = solver.C
#     ∇C = solver.∇C
#
#     # Calculate residual
#     r[1] = solver.fVal[1] - Qinv[1]*Q[1].x
#     for k = 1:N-1
#         C = ∇C[k].x[a[k],x]
#         D = ∇C[k].u[a[k],u]
#         r[2k] = fVal[k] - (∇F[k].xx*Qinv[k]*Q[k].x + ∇F[k].xu*Rinv[k]*Q[k].u - Qinv[k+1]*Q[k+1].x)
#         r[2k+1] = c[k][a[k]] - (C*Qinv[k]*Q[k].x + D*Rinv[k]*Q[k].u)
#     end
#     C = ∇C[N].x[a[N],:]
#     r[end] = c[N][a[N]] - C*Qinv[N]*Q[N].x
#
#
#     # Initial condition
#     p = p_active[1]
#     G0 = cholesky(Qinv[1])
#     λ_[1] = G0.L\r[1]
#
#     F[1] = SMatrix{n,n}(∇F[1].xx*G0.L)
#     _G = SMatrix{n,n}(∇F[1].xu*Rinv[1]*∇F[1].xu' + Qinv[2])
#     G[1] = cholesky(Symmetric(_G))
#     λ_[2] = G[1].L\(r[2] - F[1]*λ_[1])
#
#     C = ∇C[1].x[a[1],x]
#     D = ∇C[1].u[a[1],u]
#     L[1] = SMatrix{p,n}(C*G0.L)
#     M[1] = SMatrix{p,n}(D*Rinv[1]*∇F[1].xu'/(G[1].U))
#     _H = D*Rinv[1]*D' - M[1]*M[1]'
#     H[1] = cholesky(Symmetric(_H))
#     λ_[3] = H[1].L\(r[3] - M[1]*λ_[2] - L[1]*λ_[1])
#
#     G_ = G[1]
#     M_ = M[1]
#     H_ = H[1]
#     i = 4
#     for k = 2:N-1
#         # println("\n Time Step $k")
#         p = p_active[k]
#         p_ = p_active[k-1]
#
#         C = (∇C[k].x[a[k],x])
#         D = (∇C[k].u[a[k],u])
#         # C = (∇C[k].x[solver.active_set[k],:])
#         # D = (∇C[k].u[solver.active_set[k],:])
#
#         E[k] = -∇F[k].xx*Qinv[k]/G_.U
#         if p_ > 0
#             F[k] = SMatrix{n,p_}(-E[k]*M_'/H_.U)
#             G[k] = cholesky(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]' - F[k]*F[k]'))
#             λ_[i] = G[k].L\(r[i] - F[k]*λ_[i-1] - E[k]*λ_[i-2])
#         else
#             F[k] = @SMatrix zeros(n,0)
#             G[k] = cholesky(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]'))
#             λ_[i] = G[k].L\(r[i] - E[k]*λ_[i-2])
#         end
#         i += 1
#
#         if p > 0 #|| true
#             K[k] = SMatrix{p,n}(-C*Qinv[k]/G[k].U)
#             Q = solver.Q[k].xx
#             if p_ > 0
#                 L[k] = SMatrix{p,p_}(-K[k]*M_'/H_.U)
#                 _M = (C*Qinv[k]*( Q - G_.U\(I - (M_'/H_.U) * (H_.L\M_) )/G_.L )*Qinv[k]*∇F[k].xx' + D*Rinv[k]*∇F[k].xu')/G[k].U
#                 M[k] = SMatrix{p,n}(_M)
#                 H[k] = cholesky(C*Qinv[k]*C' + D*Rinv[k]*D' - K[k]*K[k]' - L[k]*L[k]' - M[k]*M[k]')
#                 λ_[i] = H[k].L\(r[i] - M[k]*λ_[i-1] - L[k]*λ_[i-2] - K[k]*λ_[i-3])
#             else
#                 L[k] = @SMatrix zeros(p,0)
#                 _M = (C*Qinv[k]*( Q - G_.U\(I)/G_.L )*Qinv[k]*∇F[k].xx' + D*Rinv[k]*∇F[k].xu')/G[k].U
#                 M[k] = SMatrix{p,n}(_M)
#                 H[k] = cholesky(C*Qinv[k]*C' + D*Rinv[k]*D' - K[k]*K[k]' - M[k]*M[k]')
#                 λ_[i] = H[k].L\(r[i] - M[k]*λ_[i-1] - K[k]*λ_[i-3])
#             end
#         else
#             L[k] = @SMatrix zeros(p_active[k],p_active[k-1])
#         end
#         i += 1
#
#         G_, M_, H_ = G[k], M[k], H[k]
#     end
#
#     # Terminal
#     p = p_active[N]
#     p_ = p_active[N-1]
#
#     C = ∇C[N].x[a[N],:]
#     D = ∇C[N].u[a[N],:]
#
#     E[N] = -C*Qinv[N]/G_.U
#     F[N] = SMatrix{p,p_}(-E[N]*M_'/H_.U)
#     G[N] = cholesky(Symmetric(C*Qinv[N]*C' - E[N]*E[N]' - F[N]*F[N]'))
#     λ_[i] = G[N].L\(r[i] - F[N]*λ_[i-1] - E[N]*λ_[i-2])
#
#     # return λ_
#
#     # BACK SUBSTITUTION
#     λ[Nb] = G[N].U\λ_[Nb]
#     λ[Nb-1] = H[N-1].U\(λ_[Nb-1] - F[N]'λ[Nb])
#     λ[Nb-2] = G[N-1].U\(λ_[Nb-2] - M[N-1]'λ[Nb-1] - E[N]'λ[Nb])
#
#     i = Nb-3
#     for k = N-2:-1:1
#         # println("\n Time step $k")
#         if p_active[k] > 0
#             if p_active[k+1] > 0
#                 λ[i] = H[k].U\(λ_[i] - F[k+1]'λ[i+1] - L[k+1]'λ[i+2])
#             else
#                 λ[i] = H[k].U\(λ_[i] - F[k+1]'λ[i+1])
#             end
#         end
#         i -= 1
#         if p_active[k] > 0
#             if p_active[k+1] > 0
#                 λ[i] = G[k].U\(λ_[i] - M[k]'λ[i+1] - E[k+1]'λ[i+2] - K[k+1]'λ[i+3])
#             else
#                 λ[i] = G[k].U\(λ_[i] - M[k]'λ[i+1] - E[k+1]'λ[i+2])
#             end
#         else
#             λ[i] = G[k].U\(λ_[i] - E[k+1]'λ[i+2] - K[k+1]'λ[i+3])
#         end
#         i -= 1
#     end
#     λ[1] = G0.U\(λ_[1] - F[1]'λ[2] - L[1]'λ[3])
#
#     return λ, λ_, r
#
# end
#
#
# function set_active_set!(solver::SequentialNewtonSolver)
#     N = length(solver.δx)
#     p = sum.(solver.active_set)
#     y_part = ones(Int,2,N-1)*n
#     y_part[2,:] = p[1:end-1]
#     y_part = vec(y_part)
#     insert!(y_part,1,3)
#     push!(y_part, p[N])
#     for i = 1:2N
#         solver.δλ[i] = zeros(y_part[i])
#         solver.r[i] = zeros(y_part[i])
#     end
# end
#
