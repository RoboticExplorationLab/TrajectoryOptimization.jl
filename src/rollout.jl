"""
$(SIGNATURES)
Simulate system dynamics for a given control trajectory U
Updates X by propagating the dynamics, using the controls specified in
U
"""
function rollout!(res::SolverVectorResults, solver::Solver)
    status = rollout!(res.X, res.U, solver)

    # Update constraints
    update_constraints!(res,solver,res.X,res.U)
    return status
end

function rollout(solver::Solver, U::Matrix)
    n,m,N = get_sizes(solver)
    X = [zeros(n) for k=1:N]
    rollout!(X, to_dvecs(U), solver)
    return to_array(X)
end

function rollout!(X::Matrix, U::Matrix, solver::Solver)
    X_vecs = to_dvecs(X)
    status = rollout!(X_vecs, to_dvecs(U), solver)
    X .= to_array(X_vecs)
    return status
end

function rollout!(X::Vector, U::Vector, solver::Solver)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    X[1] = solver.obj.x0
    for k = 1:N-1

        # Get dt is minimum time
        solver.state.minimum_time ? dt = U[k][m̄]^2 : nothing

        # Propagate dynamics forward
        solver.fd(view(X[k+1],1:n), X[k][1:n], U[k][1:m], dt)

        # Add infeasible controls
        solver.state.infeasible ? X[k+1][1:n] += U[k][m̄+1:m̄+n] : nothing

        # Check that rollout has not diverged
        if ~(norm(X[k+1],Inf) < solver.opts.max_state_value && norm(U[k],Inf) < solver.opts.max_control_value)
            return false
        end
    end

    return true
end

"""
$(SIGNATURES)
Simulate system dynamics using new control trajectory comprising
feedback gains K and feedforward gains d from backward pass
and previous control trajectory.
Line search option using alpha

flag indicates values are finite for all time steps.
"""
function rollout!(res::SolverVectorResults,solver::Solver,alpha::Float64)
    n,m,N = get_sizes(solver)
    m̄,mm = get_num_controls(solver)
    dt = solver.dt

    X = res.X; U = res.U; K = res.K; d = res.d; X_ = res.X_; U_ = res.U_

    X_[1] = solver.obj.x0;

    for k = 2:N
        # Calculate state trajectory difference
        δx = X_[k-1] - X[k-1]

        # Calculate updated control
        U_[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]

        # Get dt if minimum time
        solver.state.minimum_time ? dt = U_[k-1][m̄]^2 : nothing

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
