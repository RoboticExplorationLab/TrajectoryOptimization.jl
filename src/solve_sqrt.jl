
function chol_minus(A,B)
    AmB = LinAlg.Cholesky(copy(A),'U')
    for i = 1:size(B,1)
        LinAlg.lowrankdowndate!(AmB,B[i,:])
    end
    U = AmB[:U]
end

"""
This function is faster than other ways of checking the sign of the diagonal
"""
function checkdiag(A::AbstractArray)
    for i = 1:size(A,1)
        if A[i,i] < 0
            return false
        end
    end
    return true
end

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
