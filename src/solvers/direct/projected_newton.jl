cost(prob::Problem, V::PrimalDual) = cost(prob.obj, V.X, V.U)

############################
#       CONSTRAINTS        #
############################
function dynamics_constraints!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    N = prob.N
    X,U = V.X, V.U
    solver.fVal[1] .= V.X[1] - prob.x0
    for k = 1:N-1
         evaluate!(solver.fVal[k+1], prob.model, X[k], U[k], prob.dt)
         solver.fVal[k+1] .-= X[k+1]
     end
 end


function dynamics_jacobian!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    n,m,N = size(prob)
    X,U = V.X, V.U
    solver.∇F[1].xx .= Diagonal(I,n)
    solver.Y[1:n,1:n] .= Diagonal(I,n)
    part = (x=1:n, u =n .+ (1:m), x1=n+m .+ (1:n))
    off1 = n
    off2 = 0
    for k = 1:N-1
        jacobian!(solver.∇F[k+1], prob.model, X[k], U[k], prob.dt)
        solver.Y[off1 .+ part.x, off2 .+ part.x] .= solver.∇F[k+1].xx
        solver.Y[off1 .+ part.x, off2 .+ part.u] .= solver.∇F[k+1].xu
        solver.Y[off1 .+ part.x, off2 .+ part.x1] .= -Diagonal(I,n)
        off1 += n
        off2 += n+m
    end
end

function update_constraints!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    n,m,N = size(prob)
    for k = 1:N-1
        evaluate!(solver.C[k], prob.constraints[k], V.X[k], V.U[k])
    end
    evaluate!(solver.C[N], prob.constraints[N], V.X[N])
end

function active_set!(prob::Problem, solver::ProjectedNewtonSolver)
    n,m,N = size(prob)
    P = sum(num_constraints(prob)) + n*N
    for k = 1:N
        active_set!(solver.active_set[k], solver.C[k], solver.opts.active_set_tolerance)
    end
end

function active_set!(a::AbstractVector{Bool}, c::AbstractArray{T}, tol::T=0.0) where T
    equality, inequality = c.parts[:equality], c.parts[:inequality]
    a[equality] .= true
    a[inequality] .= c.inequality .>= -tol
end


######################################
#       CONSTRAINT JACBOBIANS        #
######################################
function constraint_jacobian!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    n,m,N = size(prob)
    for k = 1:N-1
        jacobian!(solver.∇C[k], prob.constraints[k], V.X[k], V.U[k])
    end
    jacobian!(solver.∇C[N], prob.constraints[N], V.X[N])
end

function active_constraints(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    n,m,N = size(prob)
    a = solver.a.duals
    # return view(solver.Y, a, :), view(solver.y, a)
    return solver.Y[a,:], solver.y[a]
end


############################
#      COST EXPANSION      #
############################
cost_expansion!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V) =
    cost_expansion!(solver.Q, prob.obj, V.X, V.U)


function cost_expansion!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V) where T
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    H = solver.H
    g = solver.g

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
        off += n+m
    end
    H ./= (N-1)
    g ./= (N-1)
    hess = PartedMatrix(view(H, off .+ part.x, off .+ part.x), part2)
    grad = PartedVector(view(g, off .+ part.x), part)
    hessian!(hess, prob.obj[N], V.X[N])
    gradient!(grad, prob.obj[N], V.X[N])
end



######################
#     FUNCTIONS      #
######################
function max_violation(solver::ProjectedNewtonSolver{T}) where T
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

function projection!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    Z = primals(V)
    eps_feasible = 1e-6
    count = 0
    while true
        dynamics_constraints!(prob, solver, V)
        update_constraints!(prob, solver, V)
        dynamics_jacobian!(prob, solver, V)
        constraint_jacobian!(prob, solver, V)
        active_set!(prob, solver)
        Y,y = active_constraints(prob, solver)

        viol = norm(y,Inf)
        println("feas: ", viol)
        if viol < eps_feasible || count > 10
            break
        else
            δZ = -Y'*((Y*Y')\y)
            Z .+= δZ
            count += 1
        end
    end
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

function residual(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    _,y = active_constraints(prob, solver)
    g = solver.g
    res = [g; y]
end

function newton_step!(prob::Problem, solver::ProjectedNewtonSolver)
    J0 = cost(prob)

    V = solver.V
    P = length(duals(V))
    Q = [k < N ? Expansion(prob) : Expansion(prob,:x) for k = 1:N]
    cost_expansion!(Q, prob.obj, V.X, V.U)
    dynamics_constraints!(prob, solver)
    dynamics_jacobian!(prob, solver)
    H,g = cost_expansion(Q)
    D,d = dynamics_expansion(prob, solver)

    # Form the KKT system
    A = [H D'; D zeros(P,P)]
    b = Vector([g;d])
    println("residual: ", norm(b))

    # Solve the KKT system
    δV = -A\b

    V1 = copy(V)
    V1.V .+= δV
    dynamics_constraints!(prob, solver, V1)
    dynamics_jacobian!(prob, solver, V1)
    cost_expansion!(Q, prob.obj, V1.X, V1.U)
    H,g = cost_expansion(Q)
    D,d = dynamics_expansion(prob, solver)
    b1 = [g;d]
    J = cost(prob, V1)
    println("New cost: ", J)
    println("New res: ", norm(b1))
    projection!(prob, solver, V1)


    H,g = cost_expansion(Q)
    D,d = dynamics_expansion(prob, solver)
    b2 = [g;d]
    J = cost(prob, V1)
    println("Post cost: ", J)
    println("Post res: ", norm(b2))



    α = 1

    return δV
end

function gen_usrfun_newton(prob::Problem)
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    p_colloc = n*N  # Include initial condition
    p = num_constraints(prob)
    pcum = insert!(cumsum(p), 1, 0)
    P = sum(p) + p_colloc
    part_f = create_partition2(prob.model)
    part_z = create_partition(n,m,N)
    part = (x=1:n, u=n .+ (1:m))
    part2 = (xx=(part.x,part.x), uu=(part.u,part.u),
             xu=(part.x,part.u), ux=(part.u,part.x))
    solver = ProjectedNewtonSolver(prob)

    mycost(V::Union{Primals,PrimalDual}) = cost(prob.obj, V.X, V.U)
    mycost(Z::AbstractVector) = mycost(Primals(Z, part_z))
    function grad_cost(g, V::PrimalDual)
        g_ = reshape(view(g,1:(N-1)*(n+m)), n+m, N-1)
        gterm = view(g,NN-n+1:NN)
        for k = 1:N-1
            grad = PartedArray(view(g_,:,k), part)
            gradient!(grad, prob.obj.cost[k], V.X[k], V.U[k])
        end
        g_ ./= N-1
        gradient!(gterm, prob.obj.cost[N], V.X[N])
    end
    function grad_cost(V::PrimalDual)
        g = zeros(NN)
        grad_cost(g, V)
        return g
    end

    function hess_cost(H, V::PrimalDual)
        off = 0
        for k = 1:N-1
            hess = PartedArray(view(H,off .+ (1:n+m), off .+ (1:n+m)), part2)
            hessian!(hess, prob.obj.cost[k], V.X[k], V.U[k])
            off += n+m
        end
        H ./= N-1
        hess = PartedArray(view(H,off .+ (1:n), off .+ (1:n)), part2)
        hessian!(hess, prob.obj.cost[N], V.X[N])
    end
    function hess_cost(V::PrimalDual)
        H = spzeros(NN,NN)
        hess_cost(H, V)
        return H
    end

    function dynamics(V::Union{Primals,PrimalDual})
        d = zeros(eltype(V.X[1]),p_colloc)
        dynamics(d, V)
        return d
    end
    function dynamics(d, V::Union{PrimalDual,Primals})
        d_ = reshape(d,n,N)
        d_[:,1] = V.X[1] - prob.x0
        for k = 2:N
            xdot = view(d_,:,k)
            evaluate!(xdot, prob.model, V.X[k-1], V.U[k-1], prob.dt)
            xdot .-= V.X[k]
        end
    end
    dynamics(Z::AbstractVector) = dynamics(Primals(Z, part_z))

    function jacob_dynamics(jacob, V::Union{PrimalDual,Primals})
        xdot = zeros(n)
        jacob[1:n,1:n] = Diagonal(I,n)
        off1 = n
        off2 = 0
        block = PartedMatrix(prob.model)
        jacob[1:n,1:n] = Diagonal(I,n)
        for k = 2:N
            jacobian!(block, prob.model, V.X[k-1], V.U[k-1], prob.dt)
            Jx = view(jacob, off1 .+ part.x, off2 .+ part.x)
            Ju = view(jacob, off1 .+ part.x, off2 .+ part.u)
            copyto!(Jx, block.xx)
            copyto!(Ju, block.xu)
            jacob[off1 .+ part.x, (off2+n+m) .+ part.x] = -Diagonal(I,n)
            off1 += n
            off2 += n+m
        end
    end
    function jacob_dynamics(V::Union{PrimalDual,Primals})
        jacob = spzeros(p_colloc, NN)
        jacob_dynamics(jacob, V)
        return jacob
    end


    function constraints(C, V::PrimalDual)
        update_constraints!(prob, solver, V)

        for k = 1:N
            C[pcum[k] .+ (1:p[k])] .= solver.C[k]
        end
    end
    function constraints(V::PrimalDual)
        C = zeros(sum(p))
        constraints(C, V)
        return C
    end

    function jacob_con(∇C, V::PrimalDual)

        off1 = 0
        off2 = 0
        for k = 1:N-1
            jacobian!(solver.∇C[k], prob.constraints[k], V.X[k], V.U[k])
            ∇C[off1 .+ (1:p[k]), off2 .+ (1:n+m)] .= solver.∇C[k]
            off1 += p[k]
            off2 += n+m
        end
        jacobian!(solver.∇C[N], prob.constraints[N], V.X[N])
        ∇C[off1 .+ (1:p[N]), off2 .+ (1:n)] .= solver.∇C[N]
    end
    function jacob_con(V::PrimalDual)
        ∇C = spzeros(sum(p), NN)
        jacob_con(∇C, V)
        return ∇C
    end

    function act_set(V::PrimalDual, tol=solver.opts.active_set_tolerance)
        update_constraints!(prob, solver)
        active_set!(prob, solver, V, tol)
    end


    return mycost, grad_cost, hess_cost, dynamics, jacob_dynamics, constraints, jacob_con, act_set
end


function projection2!(prob::Problem, V::PrimalDual, tol=1e-3)
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    p_colloc = n*N
    P = sum(num_constraints(prob)) + p_colloc

    V_ = copy(V)
    Z_ = primals(V_)

    mycost, grad_cost, hess_cost, dyn, jacob_dynamics, con, jacob_con, act_set =
        gen_usrfun_newton(prob)

    act_set(V_,tol)
    a = V_.active_set
    d1 = dyn(V_)
    c1 = con(V_)
    y = [d1; c1][a]
    δZ = zeros(NN)
    println("\nProjection Step:")
    println("max y: ", norm(y, Inf))
    # println("max residual: ", norm(grad_cost(V_) + jacob_dynamics(V_)'duals(V_)))
    count = 0
    while norm(y,Inf) > 1e-10
        D = jacob_dynamics(V_)
        C = jacob_con(V_)
        Y = [D; C][a,:]

        δZ = -Y'*((Y*Y')\y)
        Z_ .+= δZ

        d_ = dyn(V_)
        c_ = con(V_)
        y = [d_; c_][a]
        println("max y: ", norm(y,Inf))
        count += 1
        if count > 10
            break
        end
    end
    println("count: ", count)
end

function newton_step0(prob::Problem, V::PrimalDual, tol=1e-3)

    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    p_colloc = n*N
    P = sum(num_constraints(prob)) + p_colloc

    V_ = copy(V)
    Z_ = primals(V_)

    mycost, grad_cost, hess_cost, dyn, jacob_dynamics, con, jacob_con, act_set =
        gen_usrfun_newton(prob)

    act_set(V_,tol)
    a = V_.active_set
    d1 = dyn(V_)
    c1 = con(V_)
    y = [d1; c1][a]
    δZ = zeros(NN)
    println("\nProjection Step:")
    println("max y: ", norm(y, Inf))
    # println("max residual: ", norm(grad_cost(V_) + jacob_dynamics(V_)'duals(V_)))
    count = 0
    while norm(y,Inf) > 1e-10
        D = jacob_dynamics(V_)
        C = jacob_con(V_)
        Y = [D; C][a,:]

        δZ = -Y'*((Y*Y')\y)
        Z_ .+= δZ

        d_ = dyn(V_)
        c_ = con(V_)
        y = [d_; c_][a]
        println("max y: ", norm(y,Inf))
        count += 1
        if count > 10
            break
        end
    end
    println("count: ", count)

    J0 = mycost(V_)

    # Build and solve KKT
    act_set(V_, tol)
    a = V_.active_set
    d = dyn(V_)
    c = con(V_)
    D = jacob_dynamics(V_)
    C = jacob_con(V_)
    y = [d; c][a]
    Y = [D; C][a,:]
    g = grad_cost(V_) + Y'duals(V_)[a]
    H = hess_cost(V_)
    res0 = norm(g + Y'duals(V_)[a])

    println("\nNewton Step")
    println("Initial Cost: $J0")
    println("max y: ", norm(y, Inf))
    println("residual: ", res0)

    Pa = sum(a)
    aa = [ones(Bool, NN); a]
    A = [H Y'; Y zeros(Pa,Pa)]
    b = [g; y]
    δV = zero(V.V)
    @show cond(Array(Y*Y'))
    δV[aa] = -A\b
    err = A*δV[aa] + b
    println("err: ", norm(err))
    println("max r: ", norm(b))


    V1 = copy(V_)
    Z1 = primals(V1)
    V1.V .= V_.V + δV

    D = jacob_dynamics(V_)
    C = jacob_con(V_)
    Y = [D; C][a,:]
    res = norm(grad_cost(V1) + Y'duals(V1)[a])
    println("residual: ", res)
    println("New Cost: ", mycost(V1))
    println("max y: ", norm(dyn(V1), Inf))


    # Line search
    println("\nLine Search")
    ϕ=0.01
    α = 2
    V1 = copy(V_)
    Z1 = primals(V1)
    δV1 = α.*δV
    J = J0+1e8
    res = 1e+8
    r = Inf
    while J > J0 && r > norm(b)
        α *= 0.5
        δV1 = α.*δV
        V1.V .= V_.V + δV1

        act_set(V_, tol)
        d1 = dyn(V1)
        c1 = con(V1)
        y = [d1; c1][a]
        println("max y: ", norm(y, Inf))
        while norm(y, Inf) > 1e-6
            D = jacob_dynamics(V_)
            C = jacob_con(V_)
            Y = [D; C][a,:]
            δZ = -Y'*((Y*Y')\y)
            Z1 .+= δZ

            d1 = dyn(V1)
            c1 = con(V1)
            y1 = [d1; c1][a]
            y = y1
            println("max y: ", norm(y,Inf))
        end

        J = mycost(V1)
        print("New Cost: $J")
        res = norm(grad_cost(V1) + Y'duals(V1)[a])
        println("\t\tmax residual: ", res)
        r = norm([grad_cost(V1); d])
        println("\t\tmax r: $r")
    end
    println("α: ", α)

    # Multiplier projection
    ∇J = grad_cost(V1)
    d = dyn(V1)
    y = d
    D = jacob_dynamics(V1)
    C = jacob_con(V1)
    Y = [D; C][a,:]

    lambda = duals(V1)[a]
    res = ∇J + Y'lambda
    println("\nMultipler Projection")
    println("max y ", norm(y, Inf))
    println("max residual before: ", norm(res))
    δlambda = -(Y*Y')\(Y*res)
    lambda1 = lambda + δlambda
    r = ∇J + Y'lambda1
    println("max residual after: ", norm(r))
    V1.Y[a] .= lambda1
    J = mycost(V1)
    println("New Cost: $J")
    return V1
end

function solve(prob::Problem, opts::ProjectedNewtonSolverOptions)::Problem
    solver = ProjectedNewtonSolver(prob, opts)
    V = solver.V
    V1 = newton_step0(prob, V, opts.active_set_tolerance)
    res = copy(prob)
    copyto!(res.X, V1.X)
    copyto!(res.U, V1.U)
    # projection!(res)
    return res
end

function calc_violations(solver::Union{AugmentedLagrangianSolver{T}, ProjectedNewtonSolver{T}}) where T
    c_max = 0.0
    C = solver.C
    N = length(C)
    p = length.(C)
    v = [zeros(pi) for pi in p]
    if length(C[1]) > 0
        for k = 1:N
            v[k][C[k].parts[:equality]] = abs.(C[k].equality)
            if length(C[k].inequality) > 0
                v[k][C[k].parts[:inequality]] = max.(C[k].inequality, 0)
            end
        end
    end
    return v
end

function residual(prob::Problem, solver::ProjectedNewtonSolver)
    g = vcat([[q.x; q.u] for q in solver.Q]...)
    d = vcat(solver.fVal...)
    return norm([g;d]), norm(g), norm(d)
end


function dynamics_expansion(prob::Problem, solver::ProjectedNewtonSolver)
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    D = spzeros(N*n, NN)
    d = zeros(N*n)

    off1,off2 = n,0
    part = (x=1:n, u=n .+ (1:m), z=1:n+m, x2=n+m .+ (1:n))
    d[part.x] = solver.fVal[1]
    D[part.x, part.z] = solver.∇F[1]
    for k = 2:N
        D[off1 .+ (part.x), off2 .+ (part.z)] = solver.∇F[k]
        D[off1 .+ (part.x), off2 .+ (part.x2)] = -Diagonal(I,n)
        d[off1 .+ (part.x)] = solver.fVal[k]
        off1 += n
        off2 += n+m
    end
    return D,d
end
