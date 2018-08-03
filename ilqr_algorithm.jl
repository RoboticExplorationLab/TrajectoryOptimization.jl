include("solve_sqrt.jl")
#iLQR

"""
$(SIGNATURES)
Roll out the dynamics for a given control sequence (initial)

Updates `res.X` by propagating the dynamics, using the controls specified in
`res.U`.
"""
function rollout!(res::SolverResults,solver::Solver)
    X = res.X; U = res.U

    X[:,1] = solver.obj.x0
    for k = 1:solver.N-1
        solver.fd(view(X,:,k+1), X[:,k], U[:,k])
    end
end

"""
$(SIGNATURES)
Roll out the dynamics using the gains and optimal controls computed by the
backward pass

Updates `res.X` by propagating the dynamics at each timestep, by applying the
gains `res.K` and `res.d` to the difference between states

Will return a flag indicating if the values are finite for all time steps.
"""
function rollout!(res::SolverResults,solver::Solver,alpha::Float64)::Bool
    # pull out solver/result values
    N = solver.N
    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[:,1] = solver.obj.x0;
    for k = 2:N
        a = alpha*d[:,k-1]
        delta = X_[:,k-1] - X[:,k-1]
        U_[:, k-1] = U[:, k-1] - K[:,:,k-1]*delta - a

        solver.fd(view(X_,:,k) ,X_[:,k-1], U_[:,k-1])

        if ~all(isfinite, X_[:,k]) || ~all(isfinite, U_[:,k-1])
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)
Compute the unconstrained cost
"""
function cost(solver::Solver,X::Array{Float64,2},U::Array{Float64,2})
    # pull out solver/objective values
    N = solver.N; Q = solver.obj.Q; R = solver.obj.R; xf = solver.obj.xf; Qf = solver.obj.Qf

    J = 0.0
    for k = 1:N-1
      J += 0.5*(X[:,k] - xf)'*Q*(X[:,k] - xf) + 0.5*U[:,k]'*R*U[:,k]
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

"""
$(SIGNATURES)
Solve the dynamic programming problem, starting from the terminal time step

Computes the gain matrices K and d by applying the principle of optimality at
each time step, solving for the gradient (s) and Hessian (S) of the cost-to-go
function. Also returns parameters `v1` and `v2` (see Eq. 25a in Yuval Tassa Thesis)
"""
function backwardpass!(res::UnconstrainedResults,solver::Solver)
    N = solver.N; n = solver.model.n; m = solver.model.m; Q = solver.obj.Q; R = solver.obj.R; xf = solver.obj.xf; Qf = solver.obj.Qf

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d

    # Terminal Values
    S = Qf
    s = Qf*(X[:,N] - xf)
    v1 = 0.
    v2 = 0.

    mu = 0.
    k = N-1
    # Loop backwards
    while k >= 1
        lx = Q*(X[:,k] - xf)
        lu = R*(U[:,k])
        lxx = Q
        luu = R

        # Compute gradients of the dynamics
        fx, fu = solver.F(X[:,k],U[:,k])

        # Gradients and Hessians of Taylor Series Expansion of Q
        Qx = lx + fx'*s
        Qu = lu + fu'*s
        Qxx = lxx + fx'*S*fx
        Quu = Hermitian(luu + fu'*(S + mu*eye(n))*fu)
        Qux = fu'*(S + mu*eye(n))*fx

        # regularization
        if !isposdef(Quu)
            mu = mu + solver.opts.mu_regularization;
            k = N-1;
            if solver.opts.verbose
                println("regularized")
            end
        end

        # Compute gains
        K[:,:,k] = Quu\Qux
        d[:,k] = Quu\Qu
        s = (Qx' - Qu'*K[:,:,k] + d[:,k]'*Quu*K[:,:,k] - d[:,k]'*Qux)' # TODO: fix the transpose and simplify
        S = Qxx + K[:,:,k]'*Quu*K[:,:,k] - K[:,:,k]'*Qux - Qux'*K[:,:,k]

        # terms for line search
        v1 += float(d[:,k]'*Qu)[1]
        v2 += float(d[:,k]'*Quu*d[:,k])

        k = k - 1;
    end
    return v1, v2
end

"""
$(SIGNATURES)
Propagate dynamics with a line search
"""
function forwardpass!(res::UnconstrainedResults, solver::Solver, v1::Float64, v2::Float64)

    # pull out values from results
    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    # Compute original cost
    J_prev = cost(solver, X, U)

    J = Inf
    alpha = 1.0
    iter = 0
    dV = Inf
    z = 0.

    while z < solver.opts.c1 || z > solver.opts.c2
        flag = rollout!(res, solver, alpha)

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
        J = cost(solver, X_, U_)
        dV = alpha*v1 + (alpha^2)*v2/2.
        z = (J_prev - J)/dV[1,1]

        # Convergence criteria
        if iter > solver.opts.iterations_linesearch
            println("max iterations (forward pass)")
            break
        end

        # Reduce step size
        iter += 1
        alpha /= 2. # TODO: make this a changeable parameter
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
$(SIGNATURES)
Solve the trajectory optimization problem defined by `solver`, with `U0` as the
initial guess for the controls
"""
function solve(solver::Solver,U0::Array{Float64,2})::SolverResults
    if isa(solver.obj, UnconstrainedObjective)
        solve_unconstrained(solver, U0)
    elseif isa(solver.obj, ConstrainedObjective)
        solve_al(solver,U0)
    end
end

function solve(solver::Solver)::SolverResults
    # Generate random control sequence
    U = rand(solver.model.m, solver.N-1)
    solve(solver,U)
end


"""
$(SIGNATURES)
Solve an unconstrained optimization problem specified by `solver`
"""
function solve_unconstrained(solver::Solver,U0::Array{Float64,2})::SolverResults
    N = solver.N; n = solver.model.n; m = solver.model.m

    X = zeros(n,N)
    U = copy(U0)
    X_ = similar(X)
    U_ = similar(U)
    K = zeros(m,n,N-1)
    d = zeros(m,N-1)
    results = UnconstrainedResults(X,U,K,d,X_,U_)

    # initial roll-out
    X[:,1] = solver.obj.x0
    rollout!(results, solver)
    J_prev = cost(solver, X, U)
    if solver.opts.verbose
        println("Initial Cost: $J_prev\n")
    end

    for i = 1:solver.opts.iterations
        if solver.opts.verbose
            println("*** Iteration: $i ***")
        end
        if solver.opts.square_root
            v1, v2 = backwards_sqrt(results,solver)
        else
            v1, v2 = backwardpass!(results,solver)
        end
        J = forwardpass!(results, solver, v1, v2)
        X .= X_
        U .= U_

        if abs(J-J_prev) < solver.opts.eps
            if solver.opts.verbose
                println("-----SOLVED-----")
                println("eps criteria met at iteration: $i")
            end
            break
        end

        J_prev = copy(J)
    end

    return results
end
