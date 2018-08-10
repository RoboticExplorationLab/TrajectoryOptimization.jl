

"""
$(SIGNATURES)
Solve the dynamic programming problem, starting from the terminal time step
Computes the gain matrices K and d by applying the principle of optimality at
each time step, solving for the gradient (s) and Hessian (S) of the cost-to-go
function. Also returns parameters `v1` and `v2` (see Eq. 25a in Yuval Tassa Thesis)
"""
function backwardpass!(res::SolverIterResults,solver::Solver)
    N = solver.N; n = solver.model.n; m = solver.model.m;
    Q = solver.obj.Q; Qf = solver.obj.Qf; xf = solver.obj.xf;
    R = getR(solver)

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d

    # Terminal Values
    S = Qf
    s = Qf*(X[:,N] - xf)

    res.S[:,:,N] .= S
    res.s[:,N] = copy(s)

    v1 = 0.
    v2 = 0.
    if res isa ConstrainedResults
        C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA
        Cx = res.Cx_N
        S += Cx'*res.IμN*Cx
        s += Cx'*res.IμN*res.CN + Cx'*res.λN
    end

    mu = 0.
    k = N-1
    # Loop backwards
    while k >= 1
        lx = Q*(X[:,k] - xf)
        lu = R*(U[:,k])
        lxx = Q
        luu = R

        # Compute gradients of the dynamics
        fx,fu = res.fx[:,:,k], res.fu[:,:,k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = lx + fx'*s
        Qu = lu + fu'*s
        Qxx = lxx + fx'*S*fx
        Quu = Hermitian(luu + fu'*(S + mu*eye(n))*fu)
        # Quu = luu + fu'*(S + mu*eye(n))*fu

        Qux = fu'*(S + mu*eye(n))*fx

        # regularization
        # println("Quu: $(Quu)")
        # println("condition number: $(cond(Quu))")
        if !isposdef(Quu)
            mu = mu + solver.opts.mu_regularization;
            k = N-1
            if solver.opts.verbose
                println("regularized")
            end
            break

        end

        # Constraints
        if res isa ConstrainedResults
            # Cx, Cu = constraint_jacobian(X[:,k], U[:,k])
            Cx, Cu = res.Cx[:,:,k], res.Cu[:,:,k]
            Qx += Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k]
            Qu += Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k]
            Qxx += Cx'*Iμ[:,:,k]*Cx
            Quu += Cu'*Iμ[:,:,k]*Cu
            Qux += Cu'*Iμ[:,:,k]*Cx
        end

        # Compute gains
        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s = Qx - Qux'd[:,k]
        S = Qxx - Qux'K[:,:,k]

        res.S[:,:,k] .= S
        res.s[:,k] = copy(s)

        # terms for line search
        v1 += float(d[:,k]'*Qu)[1]
        v2 += float(d[:,k]'*Quu*d[:,k])

        k = k - 1;
    end

    return v1, v2
end

"""
$(SIGNATURES)
Perform a backwards pass with Cholesky Factorizations of the Cost-to-Go to
avoid ill-conditioning.
"""
function backwards_sqrt!(res::SolverResults,solver::Solver)

    N = solver.N
    n = solver.model.n
    m = solver.model.m
    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf

    Uq = chol(Q)
    Ur = chol(R)

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d

    # Terminal Cost-to-go
    if isa(solver.obj, ConstrainedObjective)
        Cx = res.Cx_N
        Su = chol(Qf + Cx'*res.IμN*Cx)
        s = Qf*(X[:,N] - xf) + Cx'*res.IμN*res.CN + Cx'*res.λN
    else
        Su = chol(Qf)
        s = Qf*(X[:,N] - xf)
    end

    res.S[:,:,N] .= Su # note: the sqrt root of the cost-to-go Hessian is cached
    res.s[:,N] = copy(s)

    # Initialization
    v1 = 0.
    v2 = 0.
    mu = 0.
    k = N-1

    # Backwards passes
    while k >= 1
        lx = Q*(X[:,k] - xf)
        lu = R*(U[:,k])
        lxx = Q
        luu = R
        fx, fu = solver.F(X[:,k],U[:,k])
        Qx = lx + fx'*s
        Qu = lu + fu'*s

        Wxx = qrfact!([Su*fx; Uq])
        Wuu = qrfact!([Su*fu; Ur])
        Qxu = fx'*(Su'Su)*fu

        if isa(solver.obj, ConstrainedObjective)
            # Constraints
            Iμ = res.Iμ; C = res.C; LAMBDA = res.LAMBDA;
            Cx, Cu = res.Cx[:,:,k], res.Cu[:,:,k]
            Iμ2 = sqrt.(Iμ[:,:,k])
            Qx += Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k]
            Qu += Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k]
            Qxu += Cx'*Iμ[:,:,k]*Cu

            Wxx = qrfact!([Wxx[:R]; Iμ2*Cx])
            Wuu = qrfact!([Wuu[:R]; Iμ2*Cu])
        end

        K[:,:,k] = Wuu[:R]\(Wuu[:R]'\Qxu')
        d[:,k] = Wuu[:R]\(Wuu[:R]'\Qu)

        s = Qx - Qxu*(Wuu[:R]\(Wuu[:R]'\Qu))

        try  # Regularization
            Su = chol_minus(Wxx[:R]+eye(n)*mu,Wuu[:R]'\Qxu')

        catch ex
            if ex isa LinAlg.PosDefException
                mu += 1
                k = N-1
            end
        end

        res.S[:,:,k] .= Su # note: the sqrt root of the cost-to-go Hessian is cached
        res.s[:,k] = copy(s)

        # terms for line search
        v1 += float(d[:,k]'*Qu)[1]
        v2 += float(d[:,k]'*Wuu[:R]'Wuu[:R]*d[:,k])

        k = k - 1;
    end
    return v1, v2
end

"""
$(SIGNATURES)
Perform the operation sqrt(A-B), where A and B are Symmetric Matrices
"""
function chol_minus(A::Matrix,B::Matrix)
    AmB = LinAlg.Cholesky(copy(A),'U')
    for i = 1:size(B,1)
        LinAlg.lowrankdowndate!(AmB,B[i,:])
    end
    U = AmB[:U]
end


"""
$(SIGNATURES)
Propagate dynamics with a line search (in-place)
"""
function forwardpass!(res::SolverIterResults, solver::Solver, v1::Float64, v2::Float64)

    # Pull out values from results
    X = res.X
    U = res.U
    K = res.K
    d = res.d
    X_ = res.X_
    U_ = res.U_
    # C = res.C
    # Iμ = res.Iμ
    # LAMBDA = res.LAMBDA
    # MU = res.MU

    # Compute original cost
    # J_prev = cost(solver, X, U, C, Iμ, LAMBDA)
    J_prev = cost(solver, res, X, U)

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z ≤ solver.opts.c1 || z > solver.opts.c2
        flag = rollout!(res,solver,alpha)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            if solver.opts.verbose
                println("Non-finite values in rollout")
            end
            alpha /= 2
            continue
        end

        # Calcuate cost
        if res isa ConstrainedResults
            update_constraints!(res,solver.c_fun,solver.obj.pI,X_,U_)
        end
        J = cost(solver, res, X_, U_)

        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV[1,1]
        if iter < 10
            alpha = alpha/2.
        else
            alpha = alpha/10.
        end

        if iter > solver.opts.iterations_linesearch
            if solver.opts.verbose
                println("max iterations (forward pass)")
            end
            break
        end
        iter += 1
    end

    if solver.opts.verbose
        max_c = max_violation(res)
        println("New cost: $J")
        println("- Max constraint violation: $max_c")
        println("- Expected improvement: $(dV[1])")
        println("- Actual improvement: $(J_prev-J)")
        println("- (z = $z)\n")
    end

    return J

end
