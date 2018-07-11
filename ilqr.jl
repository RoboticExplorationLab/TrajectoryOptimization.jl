

#iLQR
function rollout(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    X[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        X[:,k+1] = solver.fd(X[:,k],U[:,k])
    end
    return X
end

# function rollout(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},X_::Array{Float64,2},U_::Array{Float64,2},K::Array{Float64,3},d::Array{Float64,2},alpha::Float64)
#     X_prev = copy(X)
#     X[:,1] = solver.obj.x0
#     for k = 1:solver.N-1
#       U_[:,k] = U[:,k] - K[:,:,k]*(X[:,k] - X_prev[:,k]) - alpha*d[:,k]
#       X[:,k+1] = solver.fd(X[:,k],U_[:,k]);
#     end
#     return X, U_
# end

# function rollout!(solver::Solver, X::Array{Float64,2}, U::Array{Float64,2}, K::Array{Float64,3}, d::Array{Float64,2}, alpha::Float64, X_::Array{Float64,2}, U_::Array{Float64,2})
#     N = solver.N
#     X_[:,1] = solver.obj.x0;
#     for k = 2:N
#         a = alpha*(d[:,k-1]);
#         delta = (X_[:,k-1] - X[:,k-1])

#         U_[:, k-1] = U[:, k-1] - K[:,:,k-1]*delta - a;
#         X_[:,k] = solver.fd(X_[:,k-1], U_[:,k-1]);
#     end
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
    v1 = 0.0
    v2 = 0.0

    mu = 0.0
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
        if any(x->x < 0.0, (eigvals(Quu)))
            mu = mu + 1.0;
            k = N-1;
            println("regularized")
        end

        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)'
        S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]

        # terms for line search
        v1 += d[:,k]'*Qu
        v2 += d[:,k]'*Quu*d[:,k]

        k = k - 1;
    end
    return K, d, v1, v2
end

function forwardpass(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},K::Array{Float64,3},d::Array{Float64,2},J::Float64,v1,v2,c1::Float64=0.5,c2::Float64=0.85)
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    X_prev = copy(X)
    J_prev = copy(J)
    U_ = zeros(m,N-1)
    J = Inf
    dV = 0.0
    dJ = 0.0
    z = 0.0

    alpha = 1.0

    while J > J_prev || z < c1 || z > c2
        X[:,1] = solver.obj.x0
        for k = 1:N-1
            U_[:,k] = U[:,k] - K[:,:,k]*(X[:,k] - X_prev[:,k]) - alpha*d[:,k]
            X[:,k+1] = solver.fd(X[:,k],U_[:,k]);
        end
#         X_, U_ = rollout(solver,X,U,X_,U_,K,d,alpha)

         J = cost(solver,X,U_)
#        J = cost(solver,X_,U_)
        dV = alpha*v1 + (alpha^2)*v2/2.0
        dJ = J_prev - J
        z = dJ/dV[1]

        alpha = alpha/2.0;
    end

    println("New cost: $J")
    println("- Expected improvement: $(dV[1])")
    println("- Actual improvement: $(dJ)")
    println("- (z = $z)\n")

      return X, U_, J
#     return X_, U_, J
end

function solve(solver::Solver, iterations::Int64=100, eps::Float64=1e-3; control_init::String="random")
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    X = zeros(n,N)
    X_ = zeros(n,N)

    if control_init == "random"
        U = 1.0*rand(m,N-1)
    else
        U = zeros(m,N-1)
    end
    U_ = zeros(m,N-1)

    K = zeros(m,n,N-1)
    d = zeros(m,N-1)

    X = rollout(solver, X, U)
    J_prev = cost(solver, X, U)
    println("Initial Cost: $J_prev\n")

    for i = 1:iterations
        println("*** Iteration: $i ***")
        K, d, v1, v2 = backwardpass(solver,X,U,K,d)
        X, U, J = forwardpass(solver,X,U,K,d,J_prev,v1,v2)

        if abs(J-J_prev) < eps
          println("-----SOLVED-----")
          println("eps criteria met at iteration: $i")
          break
        end
        J_prev = copy(J)
    end

    return X, U
end