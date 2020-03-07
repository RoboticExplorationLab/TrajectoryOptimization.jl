
function rollout!(solver::iLQRSolver2{T,Q,n}, α) where {T,Q,n}
    Z = solver.Z; Z̄ = solver.Z̄
    K = solver.K; d = solver.d;

    Z̄[1].z = [solver.x0; control(Z[1])]

    temp = 0.0
	δx = solver.S[end].x
	δu = solver.S[end].u

    for k = 1:solver.N-1
        δx .= state_diff(solver.model, state(Z̄[k]), state(Z[k]))
		δu .= d[k] .* α
		mul!(δu, K[k], δx, 1.0, 1.0)
        ū = control(Z[k]) + δu
        set_control!(Z̄[k], ū)

        # Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]
        Z̄[k+1].z = [discrete_dynamics(Q, solver.model, Z̄[k]);
            control(Z[k+1])]

        temp = norm(Z̄[k+1].z)
        if temp > solver.opts.max_state_value
            return false
        end
    end
    return true
end

"Simulate the forward the dynamics open-loop"
function rollout!(solver::iLQRSolver)
    rollout!(solver.model, solver.Z, solver.x0)
    for k in eachindex(solver.Z)
        solver.Z̄[k].t = solver.Z[k].t
    end
end

function rollout!(model::AbstractModel, Z::Traj, x0)
    Z[1].z = [x0; control(Z[1])]
    for k = 2:length(Z)
        propagate_dynamics(DEFAULT_Q, model, Z[k], Z[k-1])
    end
end
