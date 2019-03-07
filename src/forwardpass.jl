"""
$(SIGNATURES)
Propagate dynamics with a line search (in-place)
"""
function forwardpass!(res::SolverIterResults, solver::Solver, Δv::Array,J_prev::Float64)
    # Pull out values from results
    X = res.X; U = res.U; X_ = res.X_; U_ = res.U_

    # update_constraints!(res,solver,X,U)
    # J_prev = cost(solver, res, X, U)

    J = Inf
    alpha = 1.0
    iter = 0
    z = -1.
    expected = 0.

    logger = current_logger()
    # print_header(logger,InnerIters) #TODO: fix, this errored out
    @logmsg InnerIters :iter value=0
    @logmsg InnerIters :cost value=J_prev
    # print_row(logger,InnerIters) #TODO: fix, same issue
    while (z ≤ solver.opts.line_search_lower_bound || z > solver.opts.line_search_upper_bound) && J >= J_prev

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            # set trajectories to original trajectory
            copyto!(X_,X)
            copyto!(U_,U)

            update_constraints!(res,solver,X_,U_)
            J = cost(solver, res, X_, U_)

            z = 0.
            alpha = 0.0
            expected = 0.

            @logmsg InnerLoop "Max iterations (forward pass)"
            regularization_update!(res,solver,:increase) # increase regularization
            res.ρ[1] += solver.opts.bp_reg_fp
            break
        end

        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout!(res,solver,alpha)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            @logmsg InnerIters "Non-finite values in rollout"
            iter += 1
            alpha /= 2.0
            continue
        end

        # Calcuate cost
        J = cost(solver, res, X_, U_)   # Unconstrained cost

        expected = -alpha*(Δv[1] + alpha*Δv[2])
        if expected > 0
            z  = (J_prev - J)/expected
        else
            @logmsg InnerIters "Non-positive expected decrease"
            z = -1
        end

        iter += 1
        alpha /= 2.0

        # Log messages
        @logmsg InnerIters :iter value=iter
        @logmsg InnerIters :α value=2*alpha
        @logmsg InnerIters :cost value=J
        @logmsg InnerIters :z value=z
        # print_row(logger,InnerIters)

    end  # forward pass loop

    if res isa ConstrainedIterResults
        @logmsg InnerLoop :c_max value=max_violation(res)
    end
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ value=J_prev-J
    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*alpha
    @logmsg InnerLoop :ρ value=res.ρ[1]

    if J > J_prev
        error("Error: Cost increased during Forward Pass")
    end

    return J
end

function rollout!(res::SolverVectorResults,solver::Solver,alpha::Float64)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    n̄,nn = get_num_states(solver)

    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[1][1:n] = solver.obj.x0;

    for k = 2:N
        # Calculate state trajectory difference
        δx = X_[k-1] - X[k-1]

        # Calculate updated control
        U_[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]

        # Get dt if minimum time and h to states
        if solver.state.minimum_time
            h = U_[k-1][m̄]
            X_[k][n̄] = h
            dt = h^2
        end

        # Propagate dynamics
        solver.fd(view(X_[k],1:n), X_[k-1][1:n], U_[k-1][1:m], dt)

        # Add infeasible controls
        solver.state.infeasible ? X_[k][1:n] += U_[k-1][m̄.+(1:n)] : nothing

        # Check that rollout has not diverged
        if ~(norm(X_[k],Inf) < solver.opts.max_state_value && norm(U_[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    # Update constraints
    update_constraints!(res,solver,X_,U_)

    return true
end
