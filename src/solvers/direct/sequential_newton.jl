
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

function size(solver::SequentialNewtonSolver)
    n,m = length(solver.δx[1]), length(solver.δu[1])
    N = length(solver.Q)
    return n,m,N
end

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
    V = PrimalDualVars(prob)
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

function set_active_set!(solver::SequentialNewtonSolver)
    N = length(solver.δx)
    p = sum.(solver.active_set)
    y_part = ones(Int,2,N-1)*n
    y_part[2,:] = p[1:end-1]
    y_part = vec(y_part)
    insert!(y_part,1,3)
    push!(y_part, p[N])
    for i = 1:2N
        solver.δλ[i] = zeros(y_part[i])
        solver.r[i] = zeros(y_part[i])
    end
end



function dynamics_jacobian!(prob::Problem, ∇F, X, U)
    n,m,N = size(prob)
    ∇F[1].xx .= Diagonal(I,n)
    for k = 1:N-1
        jacobian!(solver.∇F[k+1], prob.model, X[k], U[k], prob.dt)
    end
end
dynamics_jacobian!(prob::Problem, solver::SequentialNewtonSolver, V=solver.V) =
    dynamics_jacobian!(prob, solver.∇F, V.X, V.U)

# dynamics_jacobian!(prob::Problem, Y::KKTJacobian, V=solver.V) =
#     dynamics_jacobian!(prob, Y.∇F, V.X, V.U)


function active_set!(prob::Problem, solver::SequentialNewtonSolver)
    n,m,N = size(prob)
    for k = 1:N
        active_set!(solver.active_set[k], solver.C[k], solver.opts.active_set_tolerance)
    end
    # set_active_set!(solver)
end
# function active_set!(prob::Problem, solver::SequentialNewtonSolver, Y::KKTJacobian)
#     n,m,N = size(prob)
#     for k = 1:N
#         active_set!(Y.active_set[k], solver.C[k], solver.opts.active_set_tolerance)
#     end
# end


function active_constraints!(prob::Problem, solver::SequentialNewtonSolver, y)
    N = prob.N
    y[1] = solver.fVal[1]
    a = solver.active_set

    for k = 1:N-1
        y[2k] = solver.fVal[k+1]
        y[2k+1] = solver.C[k][a[k]]
    end
    y[end] = solver.C[N][a[N]]
    nothing
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

"""
Calculate Y'λ and store in δx and δu
"""
function jac_T_mult!(prob::Problem, solver::SequentialNewtonSolver, λ, δx, δu)
    N = prob.N
    Q = solver.Q
    ∇F = view(solver.∇F,2:N)
    ∇C = solver.∇C
    a = solver.active_set

    xi,ui = 1:n, n .+ (1:m)
    C = [∇C[k][a[k],xi] for k = 1:N]
    D = [∇C[k][a[k],ui] for k = 1:N-1]

    δx[1] = λ[1] + ∇F[1].xx'λ[2] + C[1]'λ[3]
    δu[1] =        ∇F[1].xu'λ[2] + D[1]'λ[3]
    for k = 2:N-1
        i = 2k
        δx[k] = -λ[i-2] + ∇F[k].xx'λ[i] + C[k]'λ[i+1]
        δu[k] =           ∇F[k].xu'λ[i] + D[k]'λ[i+1]
    end
    δx[end] = -λ[end-2] + C[N]'λ[end]
    nothing
end

"""
Calculate g + Y'λ, store result in δx and δu
"""
function cost_residual!(prob::Problem, solver::SequentialNewtonSolver, λ)
    N = prob.N
    Q = solver.Q

    # Calculate Y'λ
    jac_T_mult!(prob, solver, λ, solver.δx, solver.δu)
    δx, δu = solver.δx, solver.δu

    # Add g
    for k = 1:N-1
        δx[k] += Q[k].x
        δu[k] += Q[k].u
    end
    δx[N] += Q[N].x
    nothing
end

function residual!(prob::Problem, solver::SequentialNewtonSolver, V::PrimalDualVars=solver.V)
    λa = active_duals(V, solver.active_set)
    active_constraints!(prob, solver, solver.r)
    residual!(prob, solver, λa)
end

function res_norm(prob::Problem, solver::SequentialNewtonSolver)
    sqrt(norm(solver.δx)^2 + norm(solver.δu)^2 + norm(solver.r)^2)
end

"""
Calculate Y*z, store result in r, where z is split into x and u
"""
function jac_mult!(prob::Problem, solver::SequentialNewtonSolver, x, u, r)
    N = prob.N

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
    nothing
end

"""
Calculate y - Y*(H \\ g)
"""
function calc_residual!(prob::Problem, solver::SequentialNewtonSolver, r)
    N = prob.N

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
    r[1] = solver.fVal[1] - Qinv[1]*Q[1].x
    for k = 1:N-1
        C = ∇C[k].x[a[k],:]
        D = ∇C[k].u[a[k],:]
        r[2k] = fVal[k] - (∇F[k].xx*Qinv[k]*Q[k].x + ∇F[k].xu*Rinv[k]*Q[k].u - Qinv[k+1]*Q[k+1].x)
        r[2k+1] = c[k][a[k]] - (C*Qinv[k]*Q[k].x + D*Rinv[k]*Q[k].u)
    end
    C = ∇C[N].x[a[N],:]
    r[end] = c[N][a[N]] - C*Qinv[N]*Q[N].x
    nothing
end

"""
Calculates Hinv*Y'*((Y*Hinv*Y')\\y), which is the least norm solution to the problem
    min xᵀHx
     st Y*x = y
where y is the vector of active constraints and Y is the constraint jacobian
stores the result in vals.x and vals.u, with the intermediate lagrange multiplier stored in vals.λ
"""
function _projection!(prob::Problem, solver::SequentialNewtonSolver)
    n,m,N = size(prob)
    # active_constraints!(prob, solver, vals.r)
    solve_cholesky(prob, solver, solver.r)
    Qinv, Rinv = solver.Qinv, solver.Rinv
    δλ = solver.δλ
    δx = solver.δx
    δu = solver.δu
    ∇F = view(solver.∇F,2:N)
    ∇C = solver.∇C
    xi,ui = 1:n, n .+ (1:m)
    a = solver.active_set

    C = [∇C[k][a[k],xi] for k = 1:N]
    D = [∇C[k][a[k],ui] for k = 1:N-1]

    # Calculate Hinv*Y'λ
    jac_T_mult!(prob, solver, δλ, δx, δu)
    for k = 1:N-1
        δx[k] = Qinv[k]*δx[k]
        δu[k] = Rinv[k]*δu[k]
    end
    δx[N] .= Qinv[N]*δx[N]

    # δx[1] = Qinv[1]*(λ[1] + ∇F[1].xx'λ[2] + C[1]'λ[3])
    # δu[1] = Rinv[1]*(       ∇F[1].xu'λ[2] + D[1]'λ[3])
    # for k = 2:N-1
    #     i = 2k
    #     δx[k] = Qinv[k]*(-λ[i-2] + ∇F[k].xx'λ[i] + C[k]'λ[i+1])
    #     δu[k] = Rinv[k]*(          ∇F[k].xu'λ[i] + D[k]'λ[i+1])
    # end
    # δx[end] = Qinv[N]*(-λ[end-2] + C[N]'λ[end])
end

function projection!(prob::Problem, solver::SequentialNewtonSolver, V=solver.V, active_set_update=true)
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
        active_constraints!(prob, solver, solver.r)

        viol = maximum(norm.(solver.r,Inf))
        if solver.opts.verbose
            println("feas: ", viol)
        end
        if viol < eps_feasible || count > 10
            break
        else
            δx = solver.δx
            δu = solver.δu
            _projection!(prob, solver)
            for k = 1:N-1
                X[k] .-= δx[k]
                U[k] .-= δu[k]
            end
            X[N] .-= δx[N]
            count += 1
        end
    end
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


function multiplier_projection!(prob, solver::SequentialNewtonSolver, V=solver.V)
    N = prob.N
    a = solver.active_set

    # Calculate res = g + Y'λ
    λa = active_duals(V, a)
    cost_residual!(prob, solver, λa)

    # Calculate Y*res
    jac_mult!(prob, solver, solver.δx, solver.δu, solver.r)

    # Solve (Y*Y')\(Y*res)
    eyes = [I for k = 1:N]
    δλ, = solve_cholesky(prob, solver, solver.r, eyes, eyes)

    # Update Duals
    update_duals!(prob, solver, V, -1*δλ)
    residual!(prob, solver, V)
    return res_norm(prob, solver), δλ
end

function update_duals!(prob, solver::SequentialNewtonSolver, V=solver.V, δλ=solver.δλ)
    n,m,N = size(prob)
    a = solver.active_set

    V.λ[1] .+= δλ[1]
    for i = 2:2N-1
        if iseven(i)  # dynamics constraints
            V.λ[i] .+= δλ[i]
        else
            k = i ÷ 2
            λk = view(V.λ[i], a[k])
            λk .+= δλ[i]
        end
    end
    λk = view(solver.V.λ[2N], a[N])
    λk .+= δλ[end]
    return nothing
end




function solveKKT(prob::Problem, solver::SequentialNewtonSolver)
    # Calculate r = y - Y*Hinv*g
    calc_residual!(prob, solver, solver.r)

    # Solve (Y*Hinv*Y')\r
    δλ,λ_,r = solve_cholesky(prob, solver, solver.r)

    # Calculate g + Y'λ
    copyto!(solver.δλ, δλ)
    cost_residual!(prob, solver)
    δx = solver.Qinv .* solver.δx
    δu = solver.Rinv .* solver.δu
    return δx, δu, δλ
end




function newton_step!(prob::Problem, solver::SequentialNewtonSolver)
    V = solver.V
    verbose = solver.opts.verbose

    # Initial stats
    update!(prob, solver)
    J0 = cost(prob, V)
    res0 = res_norm(prob, solver)
    viol0 = max_violation(solver)

    # Projection
    verbose ? println("\nProjection:") : nothing
    projection!(prob, solver)
    update!(prob, solver)
    multiplier_projection!(prob, solver)


    # Solve KKT
    J1 = cost(prob, V)
    res1 = res_norm(prob, solver)
    viol1 = max_violation(solver)
    δx,δu,δλ = solveKKT(prob, solver)

    # Line Search
    verbose ? println("\nLine Search") : nothing
    V_ = line_search(prob, solver, δx,δu,δλ)
    J_ = cost(prob, V_)
    res_ = norm(residual(prob, solver))
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
    V = solver.V
    X,U,λ = V.X, V.U, V.λ
    J0 = cost(prob, V)
    update!(prob, solver)
    res0 = res_norm(prob, solver)
    a = solver.active_set

    V_ = copy(solver.V)
    X_,U_,λ_ = V_.X, V_.U, V_.λ
    count = 0
    solver.opts.verbose ? println("res0: $res0") : nothing
    while count < 10
        copyto!.(X_, X .- α*δx)
        copyto!.(U_, U .- α*δu)
        copyto!.(V_.λ, V.λ)
        update_duals!(prob, solver, V_, -α*δλ)

        # Calculate residual
        projection!(prob, solver, V_)
        res, = multiplier_projection!(prob, solver, V_)
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



function solve_cholesky(prob::Problem, solver::SequentialNewtonSolver, r,
        Qinv=solver.Qinv, Rinv=solver.Rinv)
    n,m,N = size(prob)
    Nb = 2N  # number of blocks
    p_active = sum.(solver.active_set)
    Pa = sum(p_active)

    x,u = SVector(1:n...), SVector(1:m...)

    # Init arrays
    S_factors = solver.L
    E = S_factors.E
    F = S_factors.F
    G = S_factors.G

    K = S_factors.K
    L = S_factors.L
    M = S_factors.M
    H = S_factors.H

    r = solver.r
    λ = solver.δλ
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
    if Qinv[1] isa UniformScaling
        G0 = cholesky(Diagonal(I,n))
    else
        G0 = cholesky(Qinv[1])
    end
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

        C = (∇C[k].x[solver.active_set[k],:])
        D = (∇C[k].u[solver.active_set[k],:])

        E[k] = -∇F[k].xx*Qinv[k]/G_.U
        F[k] = -E[k]*M_'/H_.U
        G[k] = cholesky(Symmetric(∇F[k].xx*Qinv[k]*∇F[k].xx' + ∇F[k].xu*Rinv[k]*∇F[k].xu' + Qinv[k+1] - E[k]*E[k]' - F[k]*F[k]'))
        λ_[i] = G[k].L\(r[i] - F[k]*λ_[i-1] - E[k]*λ_[i-2])
        i += 1

        K[k] = -C*Qinv[k]/G_.U
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
