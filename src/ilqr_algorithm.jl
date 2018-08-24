# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE CONTENTS:
#     SUMMARY: Forward and Backwards passes for iLQR algorithm
#
#     backwardpass!: iLQR backward pass
#     backwards_sqrt: iLQR backward pass with Cholesky Factorization of
#        Cost-to-Go
#     backwardpass_foh!: iLQR backward pass for first order hold on controls
#     chol_minus: Calculate sqrt(A-B)
#     forwardpass!: iLQR forward pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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


    v1 = 0.
    v2 = 0.
    if res isa ConstrainedResults
        C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA
        Cx = res.Cx_N
        S += Cx'*res.IμN*Cx
        s += Cx'*res.IμN*res.CN + Cx'*res.λN
    end

    res.S[:,:,N] .= S
    res.s[:,N] = copy(s)

    mu = res.mu_reg
    k = N-1
    # Loop backwards
    while k >= 1
        lx = Q*(X[:,k] - xf)
        lu = R*(U[:,k])
        lxx = Q
        luu = R

        # Compute gradients of the dynamics
        fx, fu = res.fx[:,:,k], res.fu[:,:,k]

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = lx + fx'*s
        Qu = lu + fu'*s
        Qxx = lxx + fx'*S*fx
        Quu = Hermitian(luu + fu'*(S + mu[1]*eye(n))*fu)
        Qux = fu'*(S + mu[1]*eye(n))*fx

        # regularization
        if !isposdef(Quu)
            mu[1] += solver.opts.mu_reg_update
            k = N-1
            if solver.opts.verbose
                println("regularized (normal bp)")
            end
            continue
        else
            mu[1] /= 1.6
        end

        # Constraints
        if res isa ConstrainedResults
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
        v1 += vec(d[:,k])'*vec(Qu)
        v2 += d[:,k]'*Quu*d[:,k]

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
    mu = res.mu_reg
    k = N-1

    # Backwards passes
    while k >= 1
        lx = Q*(X[:,k] - xf)
        lu = R*(U[:,k])
        lxx = Q
        luu = R
        fx, fu = solver.Fd(X[:,k],U[:,k])
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
            Su = chol_minus(Wxx[:R]+eye(n)*mu[1],Wuu[:R]'\Qxu')

        catch ex
            if ex isa LinAlg.PosDefException
                mu[1] += solver.opts.mu_reg_update
                k = N-1
                continue
            end
            #TODO add mu decrement
        end

        res.S[:,:,k] .= Su # note: the sqrt root of the cost-to-go Hessian is cached
        res.s[:,k] = copy(s)

        # terms for line search
        v1 += float(vec(d[:,k])'*vec(Qu))
        v2 += float(d[:,k]'*Wuu[:R]'Wuu[:R]*d[:,k])

        k = k - 1;
    end
    return v1, v2
end

"""
$(SIGNATURES)

Perform a backward pass for first order hold (foh) integration scheme
"""
function backwardpass_foh!(res::SolverIterResults,solver::Solver)
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    if solver.model.m != size(res.U,1)
        m += n
    end

    W = solver.obj.Q
    Wf = solver.obj.Qf
    xf = solver.obj.xf

    K = res.K
    b = res.b
    d = res.d

    dt = solver.dt

    X = res.X
    U = res.U

    R = getR(solver)

    S = zeros(n+m,n+m)
    s = zeros(n+m)

    # line search stuff
    v1 = 0.0
    v2 = 0.0

    # precompute once
    G = zeros(3*n+3*m,3*n+3*m)
    G[1:n,1:n] = W
    G[n+m+1:n+m+n,n+m+1:n+m+n] = 4.0*W
    G[2*n+2*m+1:3*n+2*m,2*n+2*m+1:3*n+2*m] = W

    G[n+1:n+m,n+1:n+m,] = R
    G[2*n+1:2*n+m,2*n+1:2*n+m] = 4.0*R
    G[3*n+1:3*n+m,3*n+1:3*n+m] = R

    g = zeros(3*n+3*m)

    # precompute most of E matrix once
    E = zeros(3*n+3*m,2*n+2*m)
    E[1:n,1:n] = eye(n)
    E[n+1:n+m,n+1:n+m] = eye(m)
    #E[n+m+1:n+m+n,:] = M
    E[2*n+m+1:2*n+m+m,n+1:n+m] = 0.5*eye(m)
    E[2*n+m+1:2*n+m+m,2*n+m+1:2*n+m+m] = 0.5*eye(m)
    E[2*n+2*m+1:2*n+2*m+n,n+m+1:n+m+n] = eye(n)
    E[3*n+2*m+1:end,2*n+m+1:end] = eye(m)

    # Boundary conditions
    S[1:n,1:n] = Wf
    s[1:n] = Wf*(X[:,N]-xf)

    if res isa ConstrainedResults
        C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA
        Cx = res.Cx_N
        S[1:n,1:n] += Cx'*res.IμN*Cx
        s[1:n] += Cx'*res.IμN*res.CN + Cx'*res.λN
    end

    k = N-1
    mu = res.mu_reg
    # Loop backwards
    while k >= 1

        # Calculate the L(x,u,y,v)
        Ac, Bc = res.Ac[:,:,k], res.Bc[:,:,k]
        Ad, Bd, Cd = res.fx[:,:,k], res.fu[:,:,k], res.fv[:,:,k]

        M = 0.25*[3*eye(n)+dt*Ac dt*Bc eye(n) zeros(n,m)]
        E[n+m+1:n+m+n,:] = M

        xm = M*[X[:,k];U[:,k];X[:,k+1];U[:,k+1]]
        um = (U[:,k] + U[:,k+1])/2.0

        g[1:n,1] = W*(X[:,k] - xf)
        g[n+1:n+m,1] = R*U[:,k]
        g[n+m+1:n+m+n,1] = 4.0*W*(xm - xf)
        g[n+m+n+1:n+m+n+m,1]= 4.0*R*um
        g[n+m+n+m+1:n+m+n+m+n,1] = W*(X[:,k+1] - xf)
        g[n+m+n+m+n+1:end,1] = R*U[:,k+1]

        H = dt/6*E'*G*E
        h = dt/6*g'*E

        # parse L(...) into H blocks
        Hx = h[1:n]
        Hu = h[n+1:n+m]
        Hy = h[n+m+1:n+m+n]
        Hv = h[2*n+m+1:2*n+2*m]

        Hxx = H[1:n,1:n]
        Huu = H[n+1:n+m,n+1:n+m]
        Hyy = H[n+m+1:n+m+n,n+m+1:n+m+n]
        Hvv = H[2*n+m+1:2*n+2*m,2*n+m+1:2*n+2*m]

        Hxu = H[1:n,n+1:n+m]
        Hxy = H[1:n,n+m+1:n+m+n]
        Hxv = H[1:n,2*n+m+1:2*n+2*m]
        Huy = H[n+1:n+m,n+m+1:n+m+n]
        Huv = H[n+1:n+m,2*n+m+1:2*n+2*m]
        Hyv = H[n+m+1:n+m+n,2*n+m+1:2*n+2*m]

        # Constraints
        if res isa ConstrainedResults
            Cx, Cu = res.Cx[:,:,k], res.Cu[:,:,k]
            Cy, Cv = res.Cx[:,:,k+1], res.Cu[:,:,k+1]

            Hx += (Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k])'
            Hu += (Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k])'
            Hy += (Cy'*Iμ[:,:,k+1]*C[:,k+1] + Cy'*LAMBDA[:,k+1])'
            Hv += (Cv'*Iμ[:,:,k+1]*C[:,k+1] + Cv'*LAMBDA[:,k+1])'

            Hxx += Cx'*Iμ[:,:,k]*Cx
            Huu += Cu'*Iμ[:,:,k]*Cu
            Hyy += Cy'*Iμ[:,:,k+1]*Cy
            Hvv += Cv'*Iμ[:,:,k+1]*Cv

            Hxu += Cx'*Iμ[:,:,k]*Cu
            #Hxy += Cx'*Iμ[:,:,k]*Cy
            #Hxv += Cx'*Iμ[:,:,k]*Cv
            #Huy += Cu'*Iμ[:,:,k]*Cy
            #Huv += Cu'*Iμ[:,:,k]*Cv
            Hyv += Cy'*Iμ[:,:,k+1]*Cv

        end

        # substitute in dynamics dx = Addx + Bddu1 + Cddu2
        Hx_ = Hx + Hy*Ad
        Hu_ = Hu + Hy*Bd
        Hv_ = Hv + Hy*Cd

        Hxx_ = Hxx + Hxy*Ad + Ad'*Hxy' + Ad'*Hyy*Ad
        Huu_ = Huu + Huy*Bd + Bd'*Huy' + Bd'*Hyy*Bd
        Hvv_ = Hvv + Hyv'*Cd + Cd'*Hyv + Cd'*Hyy*Cd
        Hxu_ = Hxu + Hxy*Bd + Ad'*Huy' + Ad'*Hyy*Bd
        Hxv_ = Hxv + Hxy*Cd + Ad'*Hyv + Ad'*Hyy*Cd
        Huv_ = Huv + Huy*Cd + Bd'*Hyv + Bd'*Hyy*Cd

        # parse (approximate) cost-to-go P

        #TODO regularize if needed, is this correct?

        Sy = s[1:n]
        Sv = s[n+1:n+m]
        Syy = S[1:n,1:n]
        Svv = S[n+1:n+m,n+1:n+m]
        Syv = S[1:n,n+1:n+m]

        Sx_ = Sy'*Ad
        Su_ = Sy'*Bd
        Sv_ = Sy'*Cd + Sv' # TODO come back and sort out transpose business

        Sxx_ = Ad'*Syy*Ad # I don't think this is regularized, similar to how Qxx is not during normal backwardpass
        Suu_ = Bd'*(Syy + mu[1]*eye(n))*Bd
        Svv_ = (Svv + mu[1]*eye(m)) + Cd'*(Syy + mu[1]*eye(n))*Cd + Cd'*Syv + Syv'*Cd
        Sxu_ = Ad'*(Syy + mu[1]*eye(n))*Bd
        Sxv_ = Ad'*(Syy + mu[1]*eye(n))*Cd + Ad'*Syv
        Suv_ = Bd'*(Syy + mu[1]*eye(n))*Cd + Bd'*Syv

        # collect terms to form Q
        Qx = Hx_ + Sx_
        Qu = Hu_ + Su_
        Qv = Hv_ + Sv_

        Qxx = Hxx_ + Sxx_
        Quu = Huu_ + Suu_
        Qvv = Hvv_ + Svv_

        Qxu = Hxu_ + Sxu_
        Qxv = Hxv_ + Sxv_
        Quv = Huv_ + Suv_

        # regularization
        # if !isposdef(Qvv)
        #     mu[1] += solver.opts.mu_reg_update
        #     k = N-1
        #     if solver.opts.verbose
        #         println("regularized")
        #     end
        #     continue
        # else
        #     mu[1] /= 2.0
        # end

        K[:,:,k+1] .= -Qvv\Qxv'
        b[:,:,k+1] .= -Qvv\Quv'
        d[:,k+1] .= -Qvv\vec(Qv)

        Qx_ = Qx + Qv*K[:,:,k+1] + d[:,k+1]'*Qxv' + d[:,k+1]'*Qvv*K[:,:,k+1]
        Qu_ = Qu + Qv*b[:,:,k+1] + d[:,k+1]'*Quv' + d[:,k+1]'*Qvv*b[:,:,k+1]
        Qxx_ = Qxx + Qxv*K[:,:,k+1] + K[:,:,k+1]'*Qxv' + K[:,:,k+1]'*Qvv*K[:,:,k+1]
        Quu_ = Quu + Quv*b[:,:,k+1] + b[:,:,k+1]'*Quv' + b[:,:,k+1]'*Qvv*b[:,:,k+1]
        Qxu_ = Qxu + K[:,:,k+1]'*Quv' + Qxv*b[:,:,k+1] + K[:,:,k+1]'*Qvv*b[:,:,k+1] # note, I'm using this instead of Qux'

        # generate (approximate) cost-to-go at timestep k
        s[1:n] = Qx_
        s[n+1:n+m] = Qu_
        S[1:n,1:n] = Qxx_
        S[n+1:n+m,n+1:n+m] = Quu_
        S[1:n,n+1:n+m] = Qxu_
        S[n+1:n+m,1:n] = Qxu_'

        # line search terms
        v1 += -vec(d[:,k+1])'*vec(Qv)
        v2 += vec(d[:,k+1])'*Qvv*vec(d[:,k+1])

        # at last time step, optimize over final control
        if k == 1
            K[:,:,1] .= -Quu_\Qxu_'
            b[:,:,1] .= zeros(m,m)
            d[:,1] .= -Quu_\vec(Qu_)

            v1 += vec(d[:,1])'*vec(Qu_)
            v2 += vec(d[:,1])'*Quu_*vec(d[:,1])
        end

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

    # Compute original cost
    if res isa ConstrainedResults
        update_constraints!(res,solver,X,U)
    end
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
            update_constraints!(res,solver,X_,U_)
        end
        J = cost(solver, res, X_, U_)
        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV

        if iter < 10
            alpha = alpha/2.
        else
            alpha = alpha/10.
        end

        if iter > solver.opts.iterations_linesearch
            # set trajectories to original trajectory
            X_ .= X
            U_ .= U
            if res isa ConstrainedResults
                update_constraints!(res,solver,X_,U_)
            end
            J = cost(solver, res, X_, U_)
            # dV = alpha*v1 + (alpha^2)*v2/2.
            z = (J_prev - J)/dV

            if solver.opts.verbose
                println("Max iterations (forward pass)\n -No improvement made")
            end
            # update regularization parameter
            res.mu_reg[1] += solver.opts.mu_reg_update
            break
        end
        iter += 1
    end

    if solver.opts.verbose
        println("New cost: $J")
        if res isa ConstrainedResults && !solver.opts.unconstrained
            max_c = max_violation(res)
            println("- Max constraint violation: $max_c")
        end
        println("- Expected improvement: $(dV[1])")
        println("- Actual improvement: $(J_prev-J)")
        println("- (z = $z)\n")
    end

    return J

end
