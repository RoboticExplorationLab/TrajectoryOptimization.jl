cost(prob::Problem, V::PrimalDual) = cost(prob.obj, V.X, V.U)


function dynamics_constraints!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    N = prob.N
    X,U = V.X, V.U
    for k = 1:N-1
         evaluate!(solver.fVal[k], prob.model, X[k], U[k], prob.dt)
         solver.fVal[k] -= X[k+1]
     end
 end


function dynamics_jacobian!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    N = prob.N
    X,U = V.X, V.U
    for k = 1:N-1
        jacobian!(solver.∇F[k], prob.model, X[k], U[k], prob.dt)
    end
end

cost_expansion!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V) =
    cost_expansion!(solver.Q, prob.obj, V.X, V.U)


function cost_expansion(prob::Problem, solver::ProjectedNewtonSolver) where T
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    H = spzeros(NN, NN)
    g = spzeros(NN)

    Q = solver.Q
    part = (x=1:n, u=n .+ (1:m))
    off = 0
    for k = 1:N-1
        H[off .+ part.x, off .+ part.x] = Q[k].xx
        H[off .+ part.x, off .+ part.u] = Q[k].ux'
        H[off .+ part.u, off .+ part.x] = Q[k].ux
        H[off .+ part.u, off .+ part.u] = Q[k].uu
        g[off .+ part.x] = Q[k].x
        g[off .+ part.u] = Q[k].u
        off += n+m
    end
    H[off .+ part.x, off .+ part.x] = Q[N].xx
    g[off .+ part.x] = Q[N].x
    return H,g
end

function residual(prob::Problem, solver::ProjectedNewtonSolver)
    g = vcat([[q.x; q.u] for q in solver.Q]...)
    d = vcat(solver.fVal...)
    return norm([g;d]), norm(g), norm(d)
end


function dynamics_expansion(prob::Problem, solver::ProjectedNewtonSolver)
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    D = spzeros((N-1)*n, NN)
    d = zeros((N-1)*n)

    off1,off2 = 0,0
    part = (x=1:n, u=n .+ (1:m), z=1:n+m, x2=n+m .+ (1:n))
    for k = 1:N-1
        D[off1 .+ (part.x), off2 .+ (part.z)] = solver.∇F[k]
        D[off1 .+ (part.x), off2 .+ (part.x2)] = -Diagonal(I,n)
        d[off1 .+ (part.x)] = solver.fVal[k]
        off1 += n
        off2 += n+m
    end
    return D,d
end

function projection!(prob::Problem, solver::ProjectedNewtonSolver, V=solver.V)
    D,d = dynamics_expansion(prob, solver)
    Z = primals(V)
    eps_feasible = 1e-6
    res = norm(d,Inf)
    if solver.opts.verbose
        println("feas: ", res)
    end
    while res > eps_feasible
        δZ = - D'*((D*D')\d)
        Z .+= δZ
        dynamics_constraints!(prob, solver, V)
        dynamics_jacobian!(prob, solver, V)
        D,d = dynamics_expansion(prob, solver)
        res = norm(d,Inf)
        if solver.opts.verbose
            println("feas: ", res)
        end
    end
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
    P = sum(num_constraints(prob)) + p_colloc
    part_f = create_partition2(prob.model)
    part_z = create_partition(n,m,N)
    part = (x=1:n, u=n .+ (1:m))
    part2 = (xx=(part.x,part.x), uu=(part.u,part.u),
             xu=(part.x,part.u), ux=(part.u,part.x))

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
        d = zeros(eltype(V.X[1]),P)
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
        jacob = spzeros(P, NN)
        jacob_dynamics(jacob, V)
        return jacob
    end

    return mycost, grad_cost, hess_cost, dynamics, jacob_dynamics
end

function newton_step0(prob::Problem, V::PrimalDual)

    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    p_colloc = n*N
    P = sum(num_constraints(prob)) + p_colloc

    V_ = copy(V)
    Z_ = primals(V_)

    mycost, grad_cost, hess_cost, dyn, jacob_dynamics = gen_usrfun_newton(prob)

    d1 = dyn(V)
    y = d1
    δz = zeros(NN)
    println("\nProjection Step:")
    println("max y: ", norm(y, Inf))
    println("max residual: ", norm(grad_cost(V_) + jacob_dynamics(V_)'duals(V_)))
    count = 0
    while norm(y,Inf) > 1e-10
        D = jacob_dynamics(V)
        Y = D

        δZ = -Y'*((Y*Y')\y)
        Z_ .+= δZ

        d1 = dyn(V_)
        y = d1
        # println("max y", norm(y,Inf))
        count += 1
    end
    println("count: ", count)

    J0 = mycost(V_)

    # Build and solve KKT
    d = dyn(V_)
    D = jacob_dynamics(V_)
    g = grad_cost(V_) + D'duals(V_)
    H = hess_cost(V_)
    res0 = norm(g + D'duals(V_))

    println("\nNewton Step")
    println("Initial Cost: $J0")
    println("max y: ", norm(y, Inf))
    println("max residual: ", res0)


    A = [H D'; D zeros(P,P)]
    b = [g; d]
    δV = -A\b
    err = A*δV + b
    println("err: ", norm(err))
    println("max r: $norm(b)")

    V1 = copy(V_)
    Z1 = primals(V1)
    V1.V .= V_.V + δV
    res = norm(grad_cost(V1) + jacob_dynamics(V1)'duals(V1))
    println("max residual: ", res)
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

        d = dyn(V1)
        y = d
        # println("max y: ", norm(y, Inf))
        while norm(y, Inf) > 1e-6
            D = jacob_dynamics(V1)
            Y = D
            δZ = -Y'*((Y*Y')\y)
            Z1 .+= δZ

            d1 = dyn(V1)
            y = d1
            # println("max y: ", norm(y,Inf))
        end

        J = mycost(V1)
        print("New Cost: $J")
        res = norm(grad_cost(V1) + jacob_dynamics(V1)'duals(V1))
        println("\t\tmax residual: ", res)
        r = norm([grad_cost(V1); d])
        println("\t\tmax r: $r")
    end
    println("α: ", α)

    # Multiplier projection
    ∇J = grad_cost(V1)
    d = dyn(V1)
    y = d
    D = jacob_dynamics(V)
    Y = D

    lambda = duals(V1)
    res = ∇J + Y'lambda
    println("\nMultipler Projection")
    println("max y ", norm(y, Inf))
    println("max residual before: ", norm(res))
    δlambda = -(Y*Y')\(Y*res)
    lambda1 = lambda + δlambda
    r = ∇J + Y'lambda1
    println("max residual after: ", norm(r))
    V1.Λ .= lambda1
    J = mycost(V1)
    println("New Cost: $J")
    return V1
end
