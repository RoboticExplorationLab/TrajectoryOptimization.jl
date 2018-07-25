
#iLQR
function rollout!(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    X[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        X[:,k+1] = solver.fd(X[:,k],U[:,k])
    end
end

function rollout!(solver::Solver, X::Array{Float64,2}, U::Array{Float64,2}, K::Array{Float64,3}, d::Array{Float64,2}, alpha::Float64, X_::Array{Float64,2}, U_::Array{Float64,2})
    N = solver.N
    X_[:,1] = solver.obj.x0;
    for k = 2:N
        a = alpha*(d[:,k-1]);
        delta = (X_[:,k-1] - X[:,k-1])

        U_[:, k-1] = U[:, k-1] - K[:,:,k-1]*delta - a;
        X_[:,k] = solver.fd(X_[:,k-1], U_[:,k-1]);

        if ~all(isfinite, X_[:,k]) || ~all(isfinite, U_[:,k-1])
            return false
        end
    end
    return true
end

# function rollout!(solver::Solver, X::Array{Float64,2}, U::Array{Float64,2}, K::Array{Float64,3}, d::Array{Float64,2}, alpha::Float64)
#     N = solver.N
#     X_ = zeros(solver.model.n, N);
#     U_ = zeros(solver.model.m, N)
#     rollout!(solver::Solver, X::Array{Float64,2}, U::Array{Float64,2}, K::Array{Float64,3}, d::Array{Float64,2},
#         alpha::Float64, X_::Array{Float64,2}, U_::Array{Float64,2})
#     return X_, U_
# end

function cost(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    N = solver.N
    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf

    J = 0.0
    for k = 1:N-1
      J += 0.5*(X[:,k] - xf)'*Q*(X[:,k] - xf) + 0.5*U[:,k]'*R*U[:,k]
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

function backwardpass(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},K::Array{Float64,3},d::Array{Float64,2})
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf

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
            mu = mu + 1.0;
            k = N-1;
            println("regularized")
        end

        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)'
        S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]

        # terms for line search
        v1 += float(d[:,k]'*Qu)[1]
        v2 += float(d[:,k]'*Quu*d[:,k])

        k = k - 1;
    end
    return K, d, v1, v2
end

function forwardpass!(X_, U_, solver::Solver, X::Array{Float64,2}, U::Array{Float64,2}, K::Array{Float64,3}, d::Array{Float64,2}, v1::Float64, v2::Float64, c1::Float64=0.01, c2::Float64=1.0)

    # Compute original cost
    J_prev = cost(solver, X, U)

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z < c1 || z > c2
        flag = rollout!(solver, X, U, K, d, alpha, X_, U_)

        # Check if rollout completed
        if ~flag
            # println("Bad X bar values")
            alpha /= 2
            continue
        end

        # Calcuate cost
        J = cost(solver, X_, U_)
        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV[1,1]

        if iter > 25
            println("max iterations (forward pass)")
            break
        end

        iter += 1
        alpha /= 2.
    end

    println("New cost: $J")
    println("- Expected improvement: $(dV[1])")
    println("- Actual improvement: $(J_prev-J)")
    println("- (z = $z)\n")

    return J

end

function solve(solver::Solver)
    U = zeros(solver.model.m, solver.N)
    solve(solver,U)
end

function solve(solver::Solver,U::Array{Float64,2},iterations::Int64=100,eps::Float64=1e-3)
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
    rollout!(solver, X, U)
    J_prev = cost(solver, X, U)
    println("Initial Cost: $J_prev\n")

    for i = 1:iterations
        println("*** Iteration: $i ***")
        K, d, v1, v2 = backwardpass(solver,X,U,K,d)
        J = forwardpass!(X_, U_, solver, X, U, K, d, v1, v2)

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
