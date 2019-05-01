"Simulate state trajectory with feedback control"
function rollout!(prob::Problem{T},solver::iLQRSolver{T},alpha::T=1.0) where T
    X = prob.X; U = prob.U
    K = solver.K; d = solver.d; X̄ = solver.X̄; Ū = solver.Ū

    X̄[1] = copy(prob.x0)

    for k = 2:prob.N
        # Calculate state trajectory difference
        δx = state_diff(X̄[k-1],X[k-1],prob,solver)

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

function rollout!(prob::Problem{T}) where T
    N = prob.N
    X = prob.X; U = prob.U

    if !all(isfinite.(prob.X[1]))
        X[1] = copy(prob.x0)
        for k = 1:N-1
            evaluate!(X[k+1], prob.model, X[k], U[k], prob.dt)
        end
    end
end

function state_diff(x̄::Vector{T},x::Vector{T},prob::Problem{T},solver::iLQRSolver{T}) where T
    if true
        x̄ - x
    else
        nothing #TODO quaternion
    end
end
