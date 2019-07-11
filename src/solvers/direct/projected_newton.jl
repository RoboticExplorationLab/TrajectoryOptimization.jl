cost(prob::Problem, V::Union{PrimalDual,PrimalDualVars}) = cost(prob.obj, V.X, V.U, get_dt_traj(prob,V.U))

############################
#       CONSTRAINTS        #
############################
function dynamics_constraints!(prob::Problem, solver::DirectSolver, V=solver.V)
    N = prob.N
    X,U,dt = V.X, V.U, get_dt_traj(prob,V.U)
    solver.fVal[1] .= V.X[1] - prob.x0
    for k = 1:N-1
         evaluate!(solver.fVal[k+1], prob.model, X[k], U[k],dt[k])
         solver.fVal[k+1] .-= X[k+1]
     end
 end


function dynamics_jacobian!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    n,m,N = size(prob)
    X,U, dt = V.X, V.U, get_dt_traj(prob,V.U)
    solver.∇F[1].xx .= Diagonal(I,n)
    solver.Y[1:n,1:n] .= Diagonal(I,n)
    part = (x=1:n, u =n .+ (1:m), x1=n+m .+ (1:n))
    p = num_constraints(prob)
    off1 = n
    off2 = 0
    for k = 1:N-1
        jacobian!(solver.∇F[k+1], prob.model, X[k], U[k], dt[k])
        solver.Y[off1 .+ part.x, off2 .+ part.x] .= solver.∇F[k+1].xx
        solver.Y[off1 .+ part.x, off2 .+ part.u] .= solver.∇F[k+1].xu
        solver.Y[off1 .+ part.x, off2 .+ part.x1] .= -Diagonal(I,n)
        off1 += n + p[k]
        off2 += n+m
    end
end


function update_constraints!(prob::Problem, solver::DirectSolver, V=solver.V)
    n,m,N = size(prob)
    for k = 1:N-1
        evaluate!(solver.C[k], prob.constraints[k], V.X[k], V.U[k])
    end
    evaluate!(solver.C[N], prob.constraints[N], V.X[N])
end

function active_set!(prob::Problem, solver::ProjectedNewtonSolver)
    n,m,N = size(prob)
    P = sum(num_constraints(prob)) + n*N
    a0 = copy(solver.a)
    for k = 1:N
        active_set!(solver.active_set[k], solver.C[k], solver.opts.active_set_tolerance)
    end
    if solver.opts.verbose && a0 != solver.a
        println("active set changed")
    end
end

function active_set!(a::AbstractVector{Bool}, c::AbstractArray{T}, tol::T=0.0) where T
    a0 = copy(a)
    equality, inequality = c.parts[:equality], c.parts[:inequality]
    a[equality] .= true
    a[inequality] .= c.inequality .>= -tol
end


######################################
#       CONSTRAINT JACBOBIANS        #
######################################
function constraint_jacobian!(prob::Problem, ∇C::Vector, X, U)
    n,m,N = size(prob)
    for k = 1:N-1
        jacobian!(∇C[k], prob.constraints[k], X[k], U[k])
    end
    jacobian!(∇C[N], prob.constraints[N], X[N])
end
constraint_jacobian!(prob::Problem, solver::DirectSolver, V=solver.V) =
    constraint_jacobian!(prob, solver.∇C, V.X, V.U)
# constraint_jacobian!(prob::Problem, Y::KKTJacobian, V=solver.V) =
#     constraint_jacobian!(prob, Y.∇C, V.X, V.U)

function active_constraints(prob::Problem, solver::ProjectedNewtonSolver)
    n,m,N = size(prob)
    a = solver.a.duals
    # return view(solver.Y, a, :), view(solver.y, a)
    return solver.Y.blocks[a,:], solver.y[a]
end


############################
#      COST EXPANSION      #
############################
function cost_expansion!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V) where T
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    H = solver.H
    g = solver.g
    dt = get_dt_traj(prob,V.U)
    part = (x=1:n, u=n .+ (1:m), z=1:n+m)
    part2 = (xx=(part.x, part.x), uu=(part.u, part.u), ux=(part.u, part.x), xu=(part.x, part.u))
    off = 0
    for k = 1:N-1
        # H[off .+ part.x, off .+ part.x] = Q[k].xx
        # H[off .+ part.x, off .+ part.u] = Q[k].ux'
        # H[off .+ part.u, off .+ part.x] = Q[k].ux
        # H[off .+ part.u, off .+ part.u] = Q[k].uu
        hess = PartedMatrix(view(H, off .+ part.z, off .+ part.z), part2)
        grad = PartedVector(view(g, off .+ part.z), part)
        hessian!(hess, prob.obj[k], V.X[k], V.U[k])
        gradient!(grad, prob.obj[k], V.X[k], V.U[k])
        hess .*= dt[k]
        grad .*= dt[k]
        off += n+m
    end
    # H .*= prob.dt
    # g .*= prob.dt
    hess = PartedMatrix(view(H, off .+ part.x, off .+ part.x), part2)
    grad = PartedVector(view(g, off .+ part.x), part)
    hessian!(hess, prob.obj[N], V.X[N])
    gradient!(grad, prob.obj[N], V.X[N])
end



######################
#     FUNCTIONS      #
######################
function update!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V, active_set=true)
    dynamics_constraints!(prob, solver, V)
    update_constraints!(prob, solver, V)
    dynamics_jacobian!(prob, solver, V)
    constraint_jacobian!(prob, solver, V)
    cost_expansion!(prob, solver, V)
    if active_set
        active_set!(prob, solver)
    end
end

function max_violation(solver::DirectSolver{T}) where T
    c_max = 0.0
    C = solver.C
    N = length(C)
    for k = 1:N
        if length(C[k].equality) > 0
            c_max = max(norm(C[k].equality,Inf), c_max)
        end
        if length(C[k].inequality) > 0
            c_max = max(pos(maximum(C[k].inequality)), c_max)
        end
        c_max = max(norm(solver.fVal[k], Inf), c_max)
    end
    return c_max
end

function calc_violations(solver::ProjectedNewtonSolver{T}) where T
    C = solver.C
    N = length(C)
    v = [zero(c) for c in C]
    v = zeros(N)
    for k = 1:N
        if length(C[k].equality) > 0
            v[k] = norm(C[k].equality,Inf)
        end
        if length(C[k].inequality) > 0
            v[k] = max(pos(maximum(C[k].inequality)), v[k])
        end
    end
    return v
end

function projection!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V, active_set_update=true)
    Z = primals(V)
    eps_feasible = solver.opts.feasibility_tolerance
    count = 0
    # cost_expansion!(prob, solver, V)
    H = Diagonal(solver.H)
    while true
        dynamics_constraints!(prob, solver, V)
        update_constraints!(prob, solver, V)
        dynamics_jacobian!(prob, solver, V)
        constraint_jacobian!(prob, solver, V)
        if active_set_update
            active_set!(prob, solver)
        end
        Y,y = active_constraints(prob, solver)
        HinvY = H\Y'

        viol = norm(y,Inf)
        if solver.opts.verbose
            println("feas: ", viol)
        end
        if viol < eps_feasible || count > 10
            break
        else
            δZ = -HinvY*((Y*HinvY)\y)
            Z .+= δZ
            count += 1
        end
    end
end

function multiplier_projection!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    g = solver.g
    a = solver.a.duals
    Y,y = active_constraints(prob, solver)
    λ = duals(V)[a]

    res0 = g + Y'λ
    δλ = -(Y*Y')\(Y*res0)
    λ_ = λ + δλ
    res = g + Y'*λ_
    copyto!(view(duals(V),a), λ_)
    res = norm(residual(prob, solver, V))
    return res, δλ
end

function solveKKT(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    a = solver.a
    δV = zero(V.V)
    λ = duals(V)[a.duals]
    Y,y = active_constraints(prob, solver)
    H,g = solver.H, solver.g
    Pa = length(y)
    A = [H Y'; Y zeros(Pa,Pa)]
    b = [g + Y'λ; y]
    δV[a] = -A\b
    return δV
end

function solveKKT_Shur(prob::Problem, solver::ProjectedNewtonSolver, Hinv, V=solver.V)
    a = solver.a
    δV = zero(V.V)
    λ = duals(V)[a.duals]
    Y,y = active_constraints(prob, solver)
    g = solver.g
    r = g + Y'λ

    YHinv = Y*Hinv
    S0 = Symmetric(YHinv*Y')
    L = cholesky(S0)
    δλ = L\(y-YHinv*r)
    δz = -Hinv*(r+Y'δλ)

    δV[solver.parts.primals] .= δz
    δV[solver.parts.duals[solver.a.duals]] .= δλ
    return δV, δλ
end

function residual(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    a = solver.a
    Y,y = active_constraints(prob, solver)
    λ = duals(V)[a.duals]
    g = solver.g
    res = [g + Y'λ; y]
end


function line_search(prob::Problem, solver::ProjectedNewtonSolver, δV)
    α = 1.0
    s = 0.01
    J0 = cost(prob, solver.V)
    update!(prob, solver)
    res0 = norm(residual(prob, solver))
    count = 0
    solver.opts.verbose ? println("res0: $res0") : nothing
    while count < 10
        V_ = solver.V + α*δV
        # projection!(prob, solver, V_)

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






function newton_step!(prob::Problem, solver::ProjectedNewtonSolver)
    V = solver.V
    verbose = solver.opts.verbose

    # Initial stats
    update!(prob, solver)
    J0 = cost(prob, V)
    res0 = norm(residual(prob, solver))
    viol0 = max_violation(solver)
    Hinv = inv(Diagonal(solver.H))

    # Projection
    verbose ? println("\nProjection:") : nothing
    projection!(prob, solver)
    update!(prob, solver)
    multiplier_projection!(prob, solver)

    # Solve KKT
    J1 = cost(prob, V)
    res1 = norm(residual(prob, solver))
    viol1 = max_violation(solver)
    δV, = solveKKT_Shur(prob, solver, Hinv)

    # Line Search
    verbose ? println("\nLine Search") : nothing
    V_ = line_search(prob, solver, δV)
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

function buildL(L::KKTFactors,y_part)
    Pa = sum(y_part)
    S = PseudoBlockArray(zeros(Pa,Pa),y_part,y_part)
    N = length(y_part) ÷ 2

    S[Block(1,1)] = L.G[end].L
    S[Block(2,1)] = L.F[1]
    S[Block(2,2)] = L.G[1].L
    S[Block(3,1)] = L.L[1]
    S[Block(3,2)] = L.M[1]
    S[Block(3,3)] = L.H[1].L
    for k = 2:N-1
        S[Block(2k,2k-2)] = L.E[k]
        S[Block(2k,2k-1)] = L.F[k]
        S[Block(2k,2k  )] = L.G[k].L
        S[Block(2k+1,2k-2)] = L.K[k]
        S[Block(2k+1,2k-1)] = L.L[k]
        S[Block(2k+1,2k-0)] = L.M[k]
        S[Block(2k+1,2k+1)] = L.H[k].L
    end
    S[Block(2N,2N-2)] = L.E[N]
    S[Block(2N,2N-1)] = L.F[N]
    S[Block(2N,2N-0)] = L.G[N].L
    return S
end
buildL(solver::SequentialNewtonSolver) = buildL(solver.L, dual_partition(solver))

function buildY(solver::SequentialNewtonSolver)
    n,m,N = size(solver)
    y_part = dual_partition(solver)
    z_part = repeat([n,m],N-1)
    push!(z_part,n)
    NN = sum(z_part)
    Pa = sum(y_part)

    # Get Active Jacobians
    xi,ui = 1:n, n .+ (1:m)
    a = solver.active_set
    C = [solver.∇C[k][a[k],1:n] for k = 1:N]
    D = [solver.∇C[k][a[k],n+1:n+m] for k = 1:N-1]

    Y = PseudoBlockArray(zeros(Pa,NN),y_part,z_part)

    Y[Block(1,1)] = Diagonal(I,n)
    for k = 1:N-1
        Y[Block(2k,2k-1)] = solver.∇F[k+1].xx
        Y[Block(2k,2k-0)] = solver.∇F[k+1].xu
        Y[Block(2k,2k+1)] = -Diagonal(I,n)
        Y[Block(2k+1,2k-1)] = C[k]
        Y[Block(2k+1,2k-0)] = D[k]
    end
    Y[Block(2N,2N-1)] = C[N]
    return Y
end
