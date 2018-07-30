using RigidBodyDynamics
using ForwardDiff
using Plots
using Base.Test

# overloaded cost function to accomodate Augmented Lagrance method
function cost(solver::Solver,X::Array{Float64,2},U::Array{Float64,2},C::Array{Float64,2},I_mu::Array{Float64,3},LAMBDA::Array{Float64,2})
    J = cost(solver,X,U)
    for k = 1:solver.N-1
        J += 0.5*(C[:,k]'*I_mu[:,:,k]*C[:,k] + LAMBDA[:,k]'*C[:,k])
    end
    return J
end

function cost(solver::Solver, res::ConstrainedResults)
    J = cost(solver,res.X,res.U)
    for k = 1:solver.N-1
        J += 0.5*(res.C[:,k]'*res.Iμ[:,:,k]*res.C[:,k] + res.LAMBDA[:,k]'*res.C[:,k])
    end
    J += 0.5*(res.CN'*res.IμN*res.CN + res.λN'*res.CN)
    return J
end

# function forwardpass!(res::ConstrainedResults, solver::Solver, v1::Float64, v2::Float64, C, Iμ, c_fun,LAMBDA::Array{Float64,2},MU::Array{Float64,2})
#     # pull out values from results
#     X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_;
#
#     # Compute original cost
#     J_prev = cost(solver, X, U, C, Iμ, LAMBDA)
#
#     pI = 2*solver.model.m # TODO change this
#
#     J = Inf
#     alpha = 1.0
#     iter = 0
#     dV = Inf
#     z = 0.
#
#     while z < solver.opts.c1 || z > solver.opts.c2
#         rollout!(res,solver,alpha)
#
#         # Calcuate cost
#         update_constraints!(C,Iμ,c_fun,X_,U_,LAMBDA,MU,pI)
#         J = cost(solver, X_, U_, C, Iμ, LAMBDA)
#         dV = alpha*v1 + (alpha^2)*v2/2.
#         z = (J_prev - J)/dV[1,1]
#         alpha = alpha/2.
#         iter = iter + 1
#
#         if iter > solver.opts.iterations_linesearch
#             if solver.opts.verbose
#                 println("max iterations (forward pass)")
#             end
#             break
#         end
#         iter += 1
#     end
#
#     if solver.opts.verbose
#         println("New cost: $J")
#         println("- Expected improvement: $(dV[1])")
#         println("- Actual improvement: $(J_prev-J)")
#         println("- (z = $z)\n")
#     end
#
#     return J
#
# end

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
    # J_prev = cost(solver, X, U, C, Iμ, LAMBDA)
    J_prev = cost(solver, res)

    pI = 2*solver.model.m # TODO change this

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z < solver.opts.c1 || z > solver.opts.c2
        rollout!(res,solver,alpha)

        # Calcuate cost
        # update_constraints!(C,Iμ,c_fun,X_,U_,LAMBDA,MU,pI)
        update_constraints!(res, c_fun, pI, X_, U_)
        # J = cost(solver, X_, U_, C, Iμ, LAMBDA)
        J = cost(solver,res)
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

    Cx, Cu = constraint_jacobian(res.X[:,N])
    S = Qf + Cx'*res.IμN*Cx
    s = Qf*(X[:,N] - xf) + + Cx'*res.IμN*res.CN + Cx'*res.λN
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

function update_constraints!(res::ConstrainedResults, c::Function, pI::Int, X::Array, U::Array)::Void
    p,N = size(res.C)
    N += 1 # since C is size p x N-1
    for k = 1:N-1
        res.C[:,k] = c(X[:,k], U[:,k])
        for j = 1:pI
            if res.C[j,k] < 0. || res.LAMBDA[j,k] > 0.
                res.Iμ[j,j,k] = res.MU[j,k]
            else
                res.Iμ[j,j,k] = 0.
            end
        end
        for j = pI+1:p
            res.Iμ[j,j,k] = res.MU[j,k]
        end
    end
    # Terminal constraint
    res.CN .= c(X[:,N])
    res.IμN .= diagm(res.μN)
    return nothing # TODO allow for more general terminal constraint
end

"""
    generate_constraint_functions(obj)
Generate the constraints function C(x,u) and a function to compute the jacobians
Cx, Cu = Jc(x,u) from a `ConstrainedObjective` type. Automatically stacks inequality
and equality constraints and takes jacobians of custom functions with `ForwardDiff`.

Stacks the constraints as follows:
[upper control inequalities
 lower control inequalities
 upper state inequalities
 lower state inequalities
 general inequalities
 general equalities
 (control equalities for infeasible start)]
"""
function generate_constraint_functions(obj::ConstrainedObjective)
    m = size(obj.R,1)
    n = length(obj.x0)

    u_min_active = isfinite.(obj.u_min)
    u_max_active = isfinite.(obj.u_max)
    x_min_active = isfinite.(obj.x_min)
    x_max_active = isfinite.(obj.x_max)

    # Inequality on control
    pI_u_max = count(u_max_active)
    pI_u = pI_u_max + count(u_min_active)
    cI_u = zeros(pI_u)
    function c_control(x,u)
        [(obj.u_max - u)[u_max_active];
         (u - obj.u_min)[u_min_active]]
    end

    # Inequality on state
    pI_x_max = count(x_max_active)
    pI_x = pI_x_max + count(x_min_active)
    function c_state(x,u)
        [(obj.x_max - x)[x_max_active];
         (x - obj.x_min)[x_min_active]]
    end

    # Custom constraints
    pI_c = obj.pI - pI_x - pI_u

    # Form inequality constraint
    CI = zeros(obj.pI)
    function cI(x,u)
        CI[1:pI_u] = c_control(x,u)
        CI[(1:pI_x)+pI_u] = c_state(x,u)
        CI[(1:pI_c)+pI_u+pI_x] = obj.cI(x,u)
        return CI
    end

    C = zeros(obj.p)
    function c_fun(x,u)
        C[1:obj.pI] = cI(x,u)
        C[1+obj.pI:end] = obj.cE(x,u)
        return C
    end

    # TODO make this more general
    function c_fun(x)
        x - obj.xf
    end

    ### Jacobians ###
    # Declare known jacobians
    fx_control = zeros(pI_u,n)
    fu_control = zeros(pI_u,m)
    fu_control[1:pI_u_max, :] = -eye(m)
    fu_control[1+pI_u_max:end,:] = eye(m)

    fx_state = zeros(pI_x,n)
    fu_state = zeros(pI_x,m)
    fx_state[1:pI_x_max, :] = -eye(pI_x)
    fx_state[1+pI_x_max:end,:] = eye(pI_x)

    fx = zeros(obj.p,n)
    fu = zeros(obj.p,m)

    fx_N = eye(n)  # Jacobian of final state

    function constraint_jacobian(x::Array,u::Array)
        fx[1:pI_u, :] = fx_control
        fu[1:pI_u, :] = fu_control
        fx[(1:pI_x)+pI_u, :] = fx_state
        fu[(1:pI_x)+pI_u, :] = fu_state
        # F_aug = F([x;u]) # TODO handle general constraints
        # fx = F_aug[:,1:n]
        # fu = F_aug[:,n+1:n+m]
        return fx, fu
    end

    function constraint_jacobian(x::Array)
        return fx_N
    end

    return c_fun, constraint_jacobian
end


function solve_al(solver::iLQR.Solver,U0::Array{Float64,2})
    N = solver.N
    n = solver.model.n
    m = solver.model.m

    ### Constraints
    p = solver.obj.p
    pI = solver.obj.pI
    u_min = solver.obj.u_min
    u_max = solver.obj.u_max
    xf = solver.obj.xf

    results = ConstrainedResults(n,m,p,N)
    results.U .= U0
    X = results.X; X_ = results.X_
    U = results.U; U_ = results.U_

    # Initialization
    # U = copy(U0)
    # X = zeros(n,N)
    # X_ = similar(X)
    # U_ = similar(U)
    # K = zeros(m,n,N-1)
    # d = zeros(m,N-1)
    # J = 0.

    # C = zeros(p,N-1)
    # Iμ = zeros(p,p,N-1)
    # LAMBDA = zeros(p,N-1)
    # MU = ones(p,N-1)
    #
    # CN = zeros(n)  # TODO allow more general terminal constraint
    # IμN = zeros(n,n)
    # λN = zeros(n)
    # μN = zeros(n)

    # results = ConstrainedResults(X,U,K,d,X_,U_,C,Iμ,LAMBDA,MU,CN,IμN,λN,μN)

    # function c_control(x,u)
    #     [u_max - u;
    #      u - u_min]
    # end
    #
    # function c_fun(x,u)
    #     c_control(x,u)
    # end
    #
    # function c_fun(x)
    #     x - xf
    # end

    # function f_augmented(f::Function, n::Int, m::Int)
    #     f_aug(S::Array) = f(S[1:n], S[n+(1:m)])
    # end
    #
    # c_aug = f_augmented(c_fun,n,m)
    # F(S) = ForwardDiff.jacobian(c_aug, S)

    # fx_control = zeros(2m,n)
    # fu_control = zeros(2m,m)
    # fu_control[1:m, :] = -eye(m)
    # fu_control[m+1:end,:] = eye(m)
    #
    # fx = zeros(p,n)
    # fu = zeros(p,m)
    #
    # fx_N = eye(n)  # Jacobian of final state
    #
    # function constraint_jacobian(x::Array,u::Array)
    #     fx[1:2m, :] = fx_control
    #     fu[1:2m, :] = fu_control
    #     # F_aug = F([x;u]) # TODO handle arbitrary constraints
    #     # fx = F_aug[:,1:n]
    #     # fu = F_aug[:,n+1:n+m]
    #     return fx, fu
    # end
    #
    # function constraint_jacobian(x::Array)
    #     return fx_N
    # end

    c_fun2, constraint_jacobian2 = generate_constraint_functions(solver.obj)

    ### SOLVER
    # initial roll-out
    X[:,1] = solver.obj.x0
    rollout!(results,solver)


    # Outer Loop
    for k = 1:solver.opts.iterations_outerloop

        # update_constraints!(C,Iμ,c_fun,X,U,LAMBDA,MU,pI)
        update_constraints!(results, c_fun2, pI, X, U)
        # J_prev = cost(solver, X, U, C, Iμ, LAMBDA)
        J_prev = cost(solver, results)
        if solver.opts.verbose
            println("Cost ($k): $J_prev\n")
        end

        for i = 1:solver.opts.iterations
            if solver.opts.verbose
                println("--Iteration: $k-($i)--")
            end
            v1, v2 = backwardpass!(results, solver, constraint_jacobian2, pI)
            J = forwardpass!(results, solver, v1, v2, c_fun2)
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
        for jj = 1:N-1
            for ii = 1:p
                results.LAMBDA[ii,jj] += -results.MU[ii,jj]*min(results.C[ii,jj],0) # TODO handle equality constraints
                results.MU[ii,jj] += 10.0
            end
        end
        # Terminal constraint
        results.λN .+= -results.μN.*results.CN
        results.μN .+= 10.0
    end

    return X, U

end
