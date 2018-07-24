
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

function backwards_sqrt(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},K::Array{Float64,3},d::Array{Float64,2})
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf
    
    Uq = chol(Q)
    Ur = chol(R)
    Su = chol(Qf)

#     S = Qf
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
        
        Wxx = qrfact!([Su*fx; Uq])
        if mu > 0
            Wuu = qrfact!([Su*fu; Ur; eye(m)*mu])
        else
            Wuu = qrfact!([Su*fu; Ur])
        end
        Qxu = fx'*(Su'Su)*fu
#         Qxx = lxx + fx'*S*fx
#         Quu = luu + fu'*(S + mu*eye(n))*fu
#         Qux = fu'*(S + mu*eye(n))*fx

        # regularization
        # println(isposdef(Wuu[:R]'Wuu[:R]))
        # if ~checkdiag(Wuu[:R])
        #     println("Regularized: mu = $mu, k=$k")
        #     println("eigs: ", eigvals(Wuu[:R]))
        #     mu = mu + 1.0;
        #     k = N-1;
        # end        
        
        K[:,:,k] = Wuu[:R]\(Wuu[:R]'\Qxu')
        d[:,k] = Wuu[:R]\(Wuu[:R]'\Qu)
#         K[:,:,k] = Quu\Qux
#         d[:,k] = Quu\Qu
        
        s = (Qx' - (Wuu[:R]'\Qu)'*(Wuu[:R]'\Qxu'))'
        Su = chol_minus(Wxx[:R],Wuu[:R]'\Qxu')
#         s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)'
#         S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]
        

        # terms for line search
        v1 += float(d[:,k]'*Qu)[1]
        v2 += float(d[:,k]'*Wuu[:R]'Wuu[:R]*d[:,k])

        k = k - 1;
    end
    return K, d, v1, v2
    
end


function solve_sqrt(solver::Solver, U::Array{Float64,2}, iterations::Int64=100, eps::Float64=1e-3)
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    X = zeros(n,N)
    X_ = similar(X)
    U_ = similar(U)
    K = zeros(m,n,N-1)
    d = zeros(m,N-1)

    X[:,1] = solver.obj.x0

    # initial roll-out
    iLQR.rollout!(solver, X, U)
    J_prev = iLQR.cost(solver, X, U)
    println("Initial Cost: $J_prev\n")

    for i = 1:iterations
        println("*** Iteration: $i ***")
        K, d, v1, v2 = backwards_sqrt(solver,X,U,K,d)
        J = iLQR.forwardpass!(X_, U_, solver, X, U, K, d, v1, v2)

        X = copy(X_)
        U = copy(U_)

        if abs(J-J_prev) < eps
            println("-----SOLVED-----")
            println("eps criteria met at iteration: $i")
            break
        end

        J_prev = copy(J)
    end

    return X, U
end