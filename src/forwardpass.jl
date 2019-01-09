"""
$(SIGNATURES)
Propagate dynamics with a line search (in-place)
"""
function forwardpass!(res::SolverIterResults, solver::Solver, Δv::Array,J_prev::Float64)
    # Pull out values from results
    X = res.X; U = res.U; X_ = res.X_; U_ = res.U_

    update_constraints!(res,solver,X,U)
    J_prev = cost(solver, res, X, U)

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
    while (z ≤ solver.opts.z_min || z > solver.opts.z_max) && J >= J_prev

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
