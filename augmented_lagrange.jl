using RigidBodyDynamics
using ForwardDiff
using Plots
using Base.Test

# overloaded cost function to accomodate Augmented Lagrance method
function cost(solver::Solver, res::ConstrainedResults, X, U)
    J = cost(solver, X, U)
    for k = 1:solver.N-1
        J += 0.5*(res.C[:,k]'*res.Iμ[:,:,k]*res.C[:,k] + res.LAMBDA[:,k]'*res.C[:,k])
    end
    J += 0.5*(res.CN'*res.IμN*res.CN + res.λN'*res.CN)
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
    # J_prev = cost(solver, X, U, C, Iμ, LAMBDA)
    J_prev = cost(solver, res, X, U)

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
        update_constraints!(res,c_fun,pI,X_,U_)
        # J = cost(solver, X_, U_, C, Iμ, LAMBDA)
        J = cost(solver, res, X_, U_)

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
        max_c = max_violation(res)
        println("New cost: $J")
        println("- constraint violation: $max_c")
        println("- Expected improvement: $(dV[1])")
        println("- Actual improvement: $(J_prev-J)")
        println("- (z = $z)\n")
    end

    return J

end

function backwardpass!(res::ConstrainedResults, solver::Solver, constraint_jacobian::Function)
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

function update_constraints!(res::ConstrainedResults, c::Function, pI::Int, X::Array, U::Array)::Void
    p,N = size(res.C)
    N += 1 # since C is size p x N-1
    for k = 1:N-1
        res.C[:,k] = c(X[:,k], U[:,k])
        for j = 1:pI
            if res.C[j,k] < 0. || res.LAMBDA[j,k] < 0.
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
function generate_constraint_functions(obj::ConstrainedObjective,infeasible::Bool=false)
    m = size(obj.R,1)
    n = length(obj.x0)

    p = obj.p
    pI = obj.pI
    pE = p-pI
    pE_c = pE  # custom equality constraints

    if infeasible
        p += n
        pE += n
    end

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
    pI_c = pI - pI_x - pI_u
    # TODO add custom constraints

    # Form inequality constraint
    CI = zeros(pI)
    function cI(x,u)
        CI[1:pI_u] = c_control(x,u)
        CI[(1:pI_x)+pI_u] = c_state(x,u)
        CI[(1:pI_c)+pI_u+pI_x] = obj.cI(x,u)
        return CI
    end

    # Augment functions together
    C = zeros(p)
    function c_fun(x,u)
        C[1:pI] = cI(x,u[1:m])
        C[(1:pE_c)+pI] = obj.cE(x,u[1:m])
        if infeasible
            C[pI+pE_c+1:end] = u[m+1:end]
        end
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

"""
    max_violation(results,inds)
Compute the maximum constraint violation. Inactive inequality constraints are
not counted (masked by the Iμ matrix). For speed, the diagonal indices can be
precomputed and passed in.
"""
function max_violation(results::ConstrainedResults,inds=CartesianIndex.(indices(results.Iμ,1),indices(results.Iμ,2)))
    maximum(abs.(results.C.*(results.Iμ[inds,:] .!= 0)))
end

function solve_al(solver::iLQR.Solver,U0::Array{Float64,2})
    N = solver.N
    n = solver.model.n
    m = solver.model.m
    J = 0.

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

    c_fun2, constraint_jacobian2 = generate_constraint_functions(solver.obj)

    # Indices for diagonal elements of Iμ matrix
    diag_inds = CartesianIndex.(indices(results.Iμ,1),indices(results.Iμ,2))


    ### SOLVER
    # initial roll-out
    X[:,1] = solver.obj.x0
    rollout!(results,solver)

    # Outer Loop
    for k = 1:solver.opts.iterations_outerloop

        update_constraints!(results,c_fun2,pI,X,U)
        J_prev = cost(solver, results, X, U)
        if solver.opts.verbose
            println("Cost ($k): $J_prev\n")
        end

        for i = 1:solver.opts.iterations
            if solver.opts.verbose
                println("--Iteration: $k-($i)--")
            end
            v1, v2 = backwardpass!(results, solver, constraint_jacobian2)
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
        outer_loop_update(results)
        if max_c < solver.opts.eps_constraint
            if solver.opts.verbose
                println("\teps constraint criteria met at outer iteration: $k\n")
            end
        end
    end

    return results

end

function outer_loop_update(results::ConstrainedResults)::Void
    p,N = size(results.C)
    N += 1
    for jj = 1:N-1
        for ii = 1:p
            results.LAMBDA[ii,jj] .+= -results.MU[ii,jj]*min(results.C[ii,jj],0)
            results.MU[ii,jj] .+= 10.0
        end
    end
    results.λN .+= results.μN.*results.CN
    results.μN .+= 10.0
    return nothing
end
