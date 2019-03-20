"""
$(SIGNATURES)
Propagate dynamics with a line search (in-place)
"""
function forwardpass!(p::Problem, res::iLQRResults, sol::iLQRSolver, ΔV::Array,J_prev::Float64)
    # Pull out values from results
    X = p.X; U = p.U; X̄ = res.X̄; Ū = res.Ū
    cost = p.cost

    J = Inf
    alpha = 1.0
    iter = 0
    z = -1.
    expected = 0.

    logger = current_logger()
    @logmsg InnerIters :iter value=0
    @logmsg InnerIters :cost value=J_prev
    while (z ≤ sol.line_search_lower_bound || z > sol.line_search_upper_bound) && J >= J_prev

        # Check that maximum number of line search decrements has not occured
        if iter > sol.iterations_linesearch
            # set trajectories to original trajectory
            copyto!(X̄,X)
            copyto!(Ū,U)

            J = cost(sol, res, X̄, Ū)

            z = 0.
            alpha = 0.0
            expected = 0.

            @logmsg InnerLoop "Max iterations (forward pass)"
            regularization_update!(res,sol,:increase) # increase regularization
            res.ρ[1] += sol.bp_reg_fp
            break
        end

        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout!(res,sol,alpha)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            @logmsg InnerIters "Non-finite values in rollout"
            iter += 1
            alpha /= 2.0
            continue
        end

        # Calcuate cost
        J = cost(sol, res, X̄, Ū)   # Unconstrained cost

        expected = -alpha*(ΔV[1] + alpha*ΔV[2])
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

    end  # forward pass loop

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
