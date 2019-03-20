function rollout!(p::Problem,res::iLQRResults,sol::iLQRSolver,alpha::Float64)
    X = p.X; U = p.U
    K = res.K; d = res.d; X̄ = res.X̄; Ū = res.Ū

    X̄[1] = p.x0

    for k = 2:p.N
        # Calculate state trajectory difference
        δx = X̄[k-1] - X[k-1]

        # Calculate updated control
        Ū[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]

        # Propagate dynamics
        p.model.fd(view(X̄[k]), X̄[k-1], Ū[k-1])

        # Check that rollout has not diverged
        if ~(norm(X̄[k],Inf) < sol.max_state_value && norm(Ū[k-1],Inf) < sol.max_control_value)
            return false
        end
    end
    return true
end

function rollout!(p::Problem)
    N = p.N
    X = p.X; U = p.U

    if isempty(X)
        for k = 1:N
            push!(X,zeros(p.model.n))
        end
    else
        X[1] = p.x0
        for k = 1:N-1
            p.model.f(X[k+1], X[k], U[k])
        end
    end
end
