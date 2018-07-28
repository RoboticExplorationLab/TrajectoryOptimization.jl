include("solve_sqrt.jl")
#iLQR
function rollout!(res::SolverResults,solver::Solver)
    X = res.X; U = res.U

    X[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        if solver.opts.inplace_dynamics
            solver.fd(view(X,:,k+1), X[:,k], U[:,k])
        else
            X[:,k+1] = solver.fd(X[:,k], U[:,k])
        end
    end
end

function rollout!(res::SolverResults,solver::Solver,alpha::Float64)
    # pull out solver/result values
    N = solver.N
    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[:,1] = solver.obj.x0;
    for k = 2:N
        a = alpha*d[:,k-1]
        delta = X_[:,k-1] - X[:,k-1]
        U_[:, k-1] = U[:, k-1] - K[:,:,k-1]*delta - a

        if solver.opts.inplace_dynamics
            solver.fd(view(X_,:,k) ,X_[:,k-1], U_[:,k-1])
        else
            X_[:,k] = solver.fd(X_[:,k-1], U_[:,k-1])
        end

        if ~all(isfinite, X_[:,k]) || ~all(isfinite, U_[:,k-1])
            return false
        end
    end
    return true
end

function cost(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    # pull out solver/objective values
    N = solver.N; Q = solver.obj.Q; R = solver.obj.R; xf = solver.obj.xf; Qf = solver.obj.Qf

    J = 0.0
    for k = 1:N-1
      J += 0.5*(X[:,k] - xf)'*Q*(X[:,k] - xf) + 0.5*U[:,k]'*R*U[:,k]
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

function backwardpass!(res::UnconstrainedResults,solver::Solver)
    N = solver.N; n = solver.model.n; m = solver.model.m; Q = solver.obj.Q; R = solver.obj.R; xf = solver.obj.xf; Qf = solver.obj.Qf

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d

    S = Qf
    s = Qf*(X[:,N] - xf)
    v1 = 0.
    v2 = 0.

    mu = 0.
    k = N-1
    while k >= 1
        lx = Q*(X[:,k] - xf)
        lu = R*(U[:,k])
        lxx = Q
        luu = R
        fx, fu = solver.F(X[:,k],U[:,k])
        Qx = lx + fx'*s
        Qu = lu + fu'*s
        Qxx = lxx + fx'*S*fx
        Quu = luu + fu'*(S + mu*eye(n))*fu
        Qux = fu'*(S + mu*eye(n))*fx

        # regularization
        if any(eigvals(Quu).<0.)
            mu = mu + solver.opts.mu_regularization;
            k = N-1;
            if solver.opts.verbose
                println("regularized")
            end
        end

        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)' # fix the transpose and simplify
        S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]

        # terms for line search
        v1 += float(d[:,k]'*Qu)[1]
        v2 += float(d[:,k]'*Quu*d[:,k])

        k = k - 1;
    end
    return v1, v2
end

function forwardpass!(res::UnconstrainedResults, solver::Solver, v1::Float64, v2::Float64)

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    # Compute original cost
    J_prev = cost(solver, X, U)

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z < solver.opts.c1 || z > solver.opts.c2
        flag = rollout!(res, solver, alpha)

        # Check if rollout completed
        if ~flag
            if solver.opts.verbose
                println("Bad X bar values")
            end
            alpha /= 2
            continue
        end

        # Calcuate cost
        J = cost(solver, X_, U_)
        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV[1,1]

        if iter > solver.opts.iterations_linesearch
            println("max iterations (forward pass)")
            break
        end

        iter += 1
        alpha /= 2.
    end

    if solver.opts.verbose
        println("New cost: $J")
        println("- Expected improvement: $(dV[1])")
        println("- Actual improvement: $(J_prev-J)")
        println("- (z = $z)\n")
    end

    return J

end

function solve(solver::Solver)
    U = zeros(solver.model.m, solver.N)
    solve(solver,U)
end

function solve(solver::Solver,U::Array{Float64,2})
    N = solver.N; n = solver.model.n; m = solver.model.m

    X = zeros(n,N)
    X_ = similar(X)
    U_ = similar(U)
    K = zeros(m,n,N-1)
    d = zeros(m,N-1)
    results = UnconstrainedResults(X,U,K,d,X_,U_)

    # initial roll-out
    X[:,1] = solver.obj.x0
    rollout!(results, solver)
    J_prev = cost(solver, X, U)
    if solver.opts.verbose
        println("Initial Cost: $J_prev\n")
    end

    for i = 1:solver.opts.iterations
        if solver.opts.verbose
            println("*** Iteration: $i ***")
        end
        if solver.opts.square_root
            K, d, v1, v2 = backwards_sqrt(solver,X,U,K,d) # TODO fix
        else
            v1, v2 = backwardpass!(results,solver)
        end
        J = forwardpass!(results, solver, v1, v2)
        X .= X_
        U .= U_

        if abs(J-J_prev) < solver.opts.eps
            if solver.opts.verbose
                println("-----SOLVED-----")
                println("eps criteria met at iteration: $i")
            end
            break
        end

        J_prev = copy(J)
    end

    return X, U
end
