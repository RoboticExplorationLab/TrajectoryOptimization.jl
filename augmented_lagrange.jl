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

function forwardpass!(X_, U_, solver::Solver, X::Array{Float64,2}, U::Array{Float64,2}, K::Array{Float64,3}, d::Array{Float64,2}, v1::Float64, v2::Float64,
        C, Iμ, c_fun,LAMBDA::Array{Float64,2},MU::Array{Float64,2}, c1::Float64=0.5, c2::Float64=0.9)

    # Compute original cost
    J_prev = cost(solver, X, U, C, Iμ, LAMBDA)

    pI = 2*solver.model.m # TODO change this

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z < c1 || z > c2
        rollout!(solver, X, U, K, d, alpha, X_, U_)

        # Calcuate cost
        update_constraints!(C,Iμ,c_fun,X_,U_,LAMBDA,MU,pI)
        J = cost(solver, X_, U_, C, Iμ, LAMBDA)
        # J = cost(solver, X_, U_, C_function(X_,U_), I_mu_function(X_,U_,LAMBDA,MU), LAMBDA)

        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV[1,1]
        alpha = alpha/2.
        iter = iter + 1

        if iter > 25
            println("max iterations (forward pass)")
            break
        end
        iter += 1
    end

    println("New cost: $J")
    println("- Expected improvement: $(dV[1])")
    println("- Actual improvement: $(J_prev-J)")
    println("- (z = $z)\n")

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
function backwardpass(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},K::Array{Float64,3},d::Array{Float64,2},
        C::Array{Float64,2}, Iμ::Array{Float64,3}, constraint_jacobian::Function, LAMBDA::Array{Float64,2})
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    Q = solver.obj.Q
    R = solver.obj.R
    xf = solver.obj.xf
    Qf = solver.obj.Qf

    # p = size(C,1)
    pI = 2*m  # Number of inequality constraints. TODO this needs to be automatic
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
        fx, fu = solver.F(X[:,k],U[:,k])
        Qx = lx + fx'*s
        Qu = lu + fu'*s
        Qxx = lxx + fx'*S*fx
        Quu = luu + fu'*(S + mu*eye(n))*fu
        Qux = fu'*(S + mu*eye(n))*fx

        Cx, Cu = constraint_jacobian(X[:,k], U[:,k])

        # regularization
        if any(eigvals(Quu).<0.)
            mu = mu + 1.0;
            k = N-1;
            println("regularized")
        end

        # Constraints

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
    return K, d, v1, v2
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


function solve_al(solver::iLQR.Solver,U::Array{Float64,2},iterations::Int64=100,eps::Float64=1e-5)
    N = solver.N
    n = solver.model.n
    m = solver.model.m
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

    function c_fun(x,u)
        [u_max - u;
         u - u_min]
    end

    function cN(x,u)
        x - xf
    end

    function f_augmented(f::Function, n::Int, m::Int)
        f_aug(S::Array) = f(S[1:n], S[n+(1:m)])
    end

    c_aug = f_augmented(c_fun,n,m)
    F(S) = ForwardDiff.jacobian(c_aug, S)

    function constraint_jacobian(x::Array,u::Array)
        F_aug = F([x;u])
        fx = F_aug[:,1:n]
        fu = F_aug[:,n+1:n+m]
        return fx, fu
    end



    # Old Stuff
    # function control_constraints(X,U)
    #     c = zeros(2*m,N)
    #
    #     for i = 1:N-1
    #         c[:,i] = [u_max - U[:,i]; U[:,i] - u_min]
    #     end
    #
    #     return c
    # end
    #
    # function control_constraints_derivatives(X,U)
    #     c = zeros(2*m,m,N)
    #
    #     for i = 1:N-1
    #         c[:,:,i] = [-eye(m); eye(m)]
    #     end
    #
    #     return c
    # end
    #
    # function terminal_state_constraints(X,U)
    #     c = zeros(size(X,1),N)
    #     c[:,N] = X[:,end] - xf
    #     return c
    # end
    #
    # function terminal_state_constraints_derivatives(X,U)
    #     c = zeros(size(X,1),size(X,1),N)
    #     c[:,:,N] = eye(size(X,1))
    #     return c
    # end
    #
    # # function C_fun(X,U)
    # #     return [control_constraints(X,U); terminal_state_constraints(X,U)]
    # # end
    #
    # function C_fun(X,U)
    #     control_constraints(X,U)
    # end
    #
    # function Cx(X,U)
    #     return [zeros(2*size(U,1),size(X,1),N); terminal_state_constraints_derivatives(X,U)]
    # end
    #
    # function Cu(X,U)
    #     return [control_constraints_derivatives(X,U); zeros(size(X,1),size(U,1),N)]
    # end
    #
    # function I_mu(X,U,LAMBDA,MU)
    #     I = zeros(m*2, m*2, N)
    #     pI = 2*m # number of inequality constraints
    #     c = C_fun(X,U)
    #     for k = 1:N
    #         # inequality constraints (controls)
    #         for j = 1:pI
    #             if c[j,k] < 0. || LAMBDA[j,k] > 0.
    #                 I[j,j,k] = MU[j,k]
    #             end
    #         end
    #
    #         # # equality constraints (terminal state constraint)
    #         # for j = pI+1:pI+n
    #         #     if k == N
    #         #         I[j,j,k] = MU[j,k]
    #         #     end
    #         # end
    #
    #     end
    #     return I
    # end

    ### SOLVER

    # initial roll-out
    X[:,1] = solver.obj.x0
    rollout!(solver, X, U)

    # Outer Loop
    for k = 1:10

        update_constraints!(C,Iμ,c_fun,X,U,LAMBDA,MU,pI)
        # J_prev = cost(solver, X, U, C_fun(X,U), I_mu(X,U,LAMBDA,MU), LAMBDA)
        J_prev = cost(solver, X, U, C, Iμ, LAMBDA)
        println("Cost ($k): $J_prev\n")

        for i = 1:iterations
            println("--Iteration: $k-($i)--")
            println(typeof(c_fun))
            K, d, v1, v2 = backwardpass(solver,X,U,K,d, C, Iμ, constraint_jacobian, LAMBDA)
            J = forwardpass!(X_, U_, solver, X, U, K, d, v1, v2, C, Iμ, c_fun, LAMBDA, MU)
            X = copy(X_)
            U = copy(U_)
            dJ = copy(abs(J-J_prev))
            J_prev = copy(J)

            if dJ < eps
                println("   eps criteria met at iteration: $i\n")
                break
            end
        end

        # Outer Loop - update lambda, mu
        println("Constraint update")
        for jj = 1:N
            for ii = 1:p
                LAMBDA[ii,jj] += -MU[ii,jj]*min(C[ii,jj],0)
                MU[ii,jj] += 10.0
            end
        end
    end

    return X, U

end
