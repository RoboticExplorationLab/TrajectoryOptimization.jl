function rollout!(prob::Problem,solver::iLQRSolver,alpha::Float64)
    X = prob.X; U = prob.U
    K = solver.K; d = solver.d; X̄ = solver.X̄; Ū = solver.Ū

    X̄[1] = prob.x0

    for k = 2:prob.N
        # Calculate state trajectory difference
        δx = X̄[k-1] - X[k-1]

        # Calculate updated control
        Ū[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]

        # Propagate dynamics
        evaluate!(X̄[k], prob.model, X̄[k-1], Ū[k-1], prob.dt)

        # Check that rollout has not diverged
        if ~(norm(X̄[k],Inf) < solver.opts.max_state_value && norm(Ū[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end
    return true
end

function rollout!(prob::Problem)
    N = prob.N
    X = prob.X; U = prob.U

    if !all(isfinite.(prob.X[1]))
        X[1] = prob.x0
        for k = 1:N-1
            evaluate!(X[k+1], prob.model, X[k], U[k], prob.dt)
        end
    end
end
