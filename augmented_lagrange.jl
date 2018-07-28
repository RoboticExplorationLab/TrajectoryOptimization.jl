using RigidBodyDynamics
using ForwardDiff
using Plots
using Base.Test

# overloaded cost function to accomodate Augmented Lagrance method
function cost(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},C::Array{Float64,2},I_mu::Array{Float64,3},LAMBDA::Array{Float64,2})
    J = cost(solver,X,U)
    for k = 1:solver.N
        J += 0.5*(C[:,k]'*I_mu[:,:,k]*C[:,k] + LAMBDA[:,k]'*C[:,k])
    end
    return J
end

function forwardpass!(res::ConstrainedResults, solver::Solver, v1::Float64, v2::Float64, C, Iμ, c_fun,LAMBDA::Array{Float64,2},MU::Array{Float64,2})
    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_;

    # Compute original cost
    J_prev = cost(solver, X, U, C, Iμ, LAMBDA)

    pI = 2*solver.model.m # TODO change this

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z < solver.opts.c1 || z > solver.opts.c2
        rollout!(res,solver,alpha)

        # Calcuate cost
        update_constraints!(C,Iμ,c_fun,X_,U_,LAMBDA,MU,pI)
        J = cost(solver, X_, U_, C, Iμ, LAMBDA)
        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV[1,1]
        alpha = alpha/2.
        iter = iter + 1

        if iter > solver.opts.iterations_linesearch
            if solver.opts.verbose
                println("max iterations (forward pass)")
            end
            break
        end
        iter += 1
    end

    if solver.opts.verbose
        println("New cost: $J")
        println("- Expected improvement: $(dV[1])")
        println("- Actual improvement: $(J_prev-J)")
        println("- (z = $z)\n")
    end

    return J

end

function forwardpass!(res::ConstrainedResults, solver::Solver, v1::Float64, v2::Float64,c_fun)

    # Pull out values from results
    X = res.X
    U = res.U
    K = res.K
    d = res.d
    X_ = res.X_
    U_ = res.U_
    C = res.C
    Iμ = res.Iμ
    LAMBDA = res.LAMBDA
    MU = res.MU

    # Compute original cost
    J_prev = cost(solver, X, U, C, Iμ, LAMBDA)

    pI = 2*solver.model.m # TODO change this

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z < solver.opts.c1 || z > solver.opts.c2
        rollout!(res,solver,alpha)

        # Calcuate cost
        update_constraints!(C,Iμ,c_fun,X_,U_,LAMBDA,MU,pI)
        J = cost(solver, X_, U_, C, Iμ, LAMBDA)
        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV[1,1]
        alpha = alpha/2.
        iter = iter + 1

        if iter > solver.opts.iterations_linesearch
            if solver.opts.verbose
                println("max iterations (forward pass)")
            end
            break
        end
        iter += 1
    end

    if solver.opts.verbose
        println("New cost: $J")
        println("- Expected improvement: $(dV[1])")
        println("- Actual improvement: $(J_prev-J)")
        println("- (z = $z)\n")
    end

    return J

end

"""
    I_mu(cI,cE,λ,μ)
Build the a diagonal matrix of μ's that accurately account for inequalities
for a single time step.

# Arguments
- cI: vector of inequality constraint values
- λ: lagrange multipliers
- μ: penalty terms

TODO: account for equality constraints
TODO: update in-place (only change the values that need to be changed)
"""
function build_I_mu(cI,λ,μ)
    p = length(λ)  # number of constraints
    pI = length(cI)  # number of inequality constraints
    pE = p-pI
    I = zeros(p, p)
    # inequality constraints (controls)
    for j = 1:pI
        if cI[j] < 0. || λ[j] > 0.
            I[j,j] = μ[j]
        end
    end

    # # equality constraints (terminal state constraint)
    # for j = pI+1:p
    #     if k == N
    #         I[j,j,k] = μ[j]
    #     end
    # end
    return I
end

# Augmented Lagrange
# function backwardpass(res::ConstrainedResults,solver::Solver, C::Array{Float64,2}, Iμ::Array{Float64,3}, constraint_jacobian::Function, LAMBDA::Array{Float64,2})
#     N = solver.N
#     n = solver.model.n
#     m = solver.model.m
#     Q = solver.obj.Q
#     R = solver.obj.R
#     xf = solver.obj.xf
#     Qf = solver.obj.Qf
#
#     # pull out values from results
#     X = res.X; U = res.U; K = res.K; d = res.d
#
#     # p = size(C,1)
#     pI = 2*m  # Number of inequality constraints. TODO this needs to be automatic
#     # pE = n
#
#     S = Qf
#     s = Qf*(X[:,N] - xf)
#     v1 = 0.
#     v2 = 0.
#
#     mu = 0.
#     k = N-1
#     while k >= 1
#         lx = Q*(X[:,k] - xf)
#         lu = R*(U[:,k])
#         lxx = Q
#         luu = R
#         fx, fu = solver.F(X[:,k],U[:,k])
#         Qx = lx + fx'*s
#         Qu = lu + fu'*s
#         Qxx = lxx + fx'*S*fx
#         Quu = luu + fu'*(S + mu*eye(n))*fu
#         Qux = fu'*(S + mu*eye(n))*fx
#
#         Cx, Cu = constraint_jacobian(X[:,k], U[:,k])
#
#         # regularization
#         if any(eigvals(Quu).<0.)
#             mu = mu + solver.opts.mu_regularization
#             k = N-1
#             if solver.opts.verbose
#                 println("regularized")
#             end
#         end
#
#         # Constraints
#
#         Qx += Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k]
#         Qu += Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k]
#         Qxx += Cx'*Iμ[:,:,k]*Cx
#         Quu += Cu'*Iμ[:,:,k]*Cu
#         Qux += Cu'*Iμ[:,:,k]*Cx
#
#         K[:,:,k] = Quu\Qux
#         d[:,k] = Quu\Qu
#         s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)'
#         S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]
#
#         # terms for line search
#         v1 += float(d[:,k]'*Qu)[1]
#         v2 += float(d[:,k]'*Quu*d[:,k])
#
#         k = k - 1;
#     end
#     return v1, v2
# end

function backwardpass!(res::ConstrainedResults, solver::Solver, constraint_jacobian::Function, pI::Int)
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; C = res.C; Iμ = res.Iμ; LAMBDA = res.LAMBDA
    # p = size(C,1)
    # pI = 2*m  # Number of inequality constraints. TODO this needs to be automatic
    # pE = n

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
        fx, fu = solver.F(X[:,k], U[:,k])
        Qx = lx + fx'*s
        Qu = lu + fu'*s
        Qxx = lxx + fx'*S*fx
        Quu = Hermitian(luu + fu'*(S + mu*eye(n))*fu)
        Qux = fu'*(S + mu*eye(n))*fx

        # regularization
        if ~isposdef(Quu)
            mu = mu + solver.opts.mu_regularization;
            k = N-1
            if solver.opts.verbose
                println("regularized")
            end
        end

        # Constraints
        Cx, Cu = constraint_jacobian(X[:,k], U[:,k])
        Qx += Cx'*Iμ[:,:,k]*C[:,k] + Cx'*LAMBDA[:,k]
        Qu += Cu'*Iμ[:,:,k]*C[:,k] + Cu'*LAMBDA[:,k]
        Qxx += Cx'*Iμ[:,:,k]*Cx
        Quu += Cu'*Iμ[:,:,k]*Cu
        Qux += Cu'*Iμ[:,:,k]*Cx
        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)'
        S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]

        # terms for line search
        v1 += float(d[:,k]'*Qu)[1]
        v2 += float(d[:,k]'*Quu*d[:,k])

        k = k - 1;
    end
    return v1, v2
end

function update_constraints!(C,Iμ,c,X,U,LAMBDA,MU,pI)
    p,N = size(C)
    for k = 1:N-1
        C[:,k] = c(X[:,k], U[:,k])
        for j = 1:pI
            if C[j,k] < 0. || LAMBDA[j,k] > 0.
                Iμ[j,j,k] = MU[j,k]
            else
                Iμ[j,j,k] = 0.
            end
        end
        for j = pI+1:p
            Iμ[j,j,k] = MU[j,k]
        end
    end
end


function solve_al(solver::iLQR.Solver,U0::Array{Float64,2})
    N = solver.N
    n = solver.model.n
    m = solver.model.m

    U = copy(U0)
    X = zeros(n,N)
    X_ = similar(X)
    U_ = similar(U)
    K = zeros(m,n,N-1)
    d = zeros(m,N-1)
    J = 0.

    ### Constraints
    p = 2*m
    pI = p
    C = zeros(p,N)
    Iμ = zeros(p,p,N)
    LAMBDA = zeros(p,N)
    MU = ones(p,N)
    u_min = -2.
    u_max = 2.
    xf = solver.obj.xf

    # results = ConstrainedResults(n,m,p,N)
    results = ConstrainedResults(X,U,K,d,X_,U_,C,Iμ,LAMBDA,MU)

    function c_control(x,u)
        [u_max - u;
         u - u_min]
    end

    function c_fun(x,u)
        c_control(x,u)
    end

    function cN(x,u)
        x - xf
    end

    # function f_augmented(f::Function, n::Int, m::Int)
    #     f_aug(S::Array) = f(S[1:n], S[n+(1:m)])
    # end
    #
    # c_aug = f_augmented(c_fun,n,m)
    # F(S) = ForwardDiff.jacobian(c_aug, S)

    fx_control = zeros(2m,n)
    fu_control = zeros(2m,m)
    fu_control[1:m, :] = -eye(m)
    fu_control[m+1:end,:] = eye(m)

    fx = zeros(p,n)
    fu = zeros(p,m)

    function constraint_jacobian(x::Array,u::Array)
        fx[1:2m, :] = fx_control
        fu[1:2m, :] = fu_control
        # F_aug = F([x;u]) # TODO handle arbitrary constraints
        # fx = F_aug[:,1:n]
        # fu = F_aug[:,n+1:n+m]
        return fx, fu
    end


    ### SOLVER
    # initial roll-out
    X[:,1] = solver.obj.x0
    rollout!(results,solver)


    # Outer Loop
    for k = 1:solver.opts.iterations_outerloop

        update_constraints!(C,Iμ,c_fun,X,U,LAMBDA,MU,pI)
        J_prev = cost(solver, X, U, C, Iμ, LAMBDA)
        if solver.opts.verbose
            println("Cost ($k): $J_prev\n")
        end

        for i = 1:solver.opts.iterations
            if solver.opts.verbose
                println("--Iteration: $k-($i)--")
            end
            v1, v2 = backwardpass!(results, solver, constraint_jacobian, pI)
            J = forwardpass!(results, solver, v1, v2, c_fun)
            X .= X_
            U .= U_
            dJ = copy(abs(J-J_prev))
            J_prev = copy(J)

            if dJ < solver.opts.eps
                if solver.opts.verbose
                    println("   eps criteria met at iteration: $i\n")
                end
                break
            end
        end

        # Outer Loop - update lambda, mu
        for jj = 1:N
            for ii = 1:p
                LAMBDA[ii,jj] += -MU[ii,jj]*min(C[ii,jj],0)
                MU[ii,jj] += 10.0
            end
        end
    end

    return X, U

end
