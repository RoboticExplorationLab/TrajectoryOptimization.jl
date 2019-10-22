
# Generic solve methods
"iLQR solve method"
function solve!(prob::StaticProblem, solver::StaticiLQRSolver{T}) where T<:AbstractFloat
    reset!(solver)
    to = solver.stats[:timer]

    n,m,N = size(prob)
    J = Inf

    logger = default_logger(solver)

    # Initial rollout
    rollout!(prob)
    live_plotting(prob,solver)

    J_prev = cost(prob)


    with_logger(logger) do
        record_iteration!(prob, solver, J_prev, Inf)
        for i = 1:solver.opts.iterations
            J = step!(prob, solver, J_prev)

            # check for cost blow up
            if J > solver.opts.max_cost_value
                @warn "Cost exceeded maximum cost"
                return solver
            end

            @timeit to "copy" begin
                copyto!(prob.X, solver.X̄)
                copyto!(prob.U, solver.Ū)
            end

            dJ = abs(J - J_prev)
            J_prev = copy(J)
            record_iteration!(prob, solver, J, dJ)
            live_plotting(prob,solver)

            println(logger, InnerLoop)
            @timeit to "convergence" evaluate_convergence(solver) ? break : nothing
        end
    end
    return solver
end

function step!(prob::Problem{T}, solver::StaticiLQRSolver{T}, J::T) where T
    to = solver.stats[:timer]
    @timeit to "jacobians" jacobian!(prob,solver)   # TODO: rename this to dynamics jacobian
    @timeit to "cost expansion" cost_expansion!(prob,solver)
    @timeit to "backward pass" ΔV = backwardpass!(prob,solver)
    @timeit to "forward pass" forwardpass!(prob,solver,ΔV,J)
end

function cost_expansion!(prob::Problem{T}, solver::StaticiLQRSolver{T}) where T
    E = solver.Q
    N = prob.N
    X,U,Dt = prob.X, prob.U, get_dt_traj(prob)
    for k in eachindex(U)
        cost_expansion!(E[k], prob.obj[k], X[k], U[k], Dt[k])
    end
    cost_expansion!(E[N], prob.obj[N], X[N])
    nothing
end

function backwardpass!(prob::StaticProblem, solver::StaticiLQRSolver)
    n,m,N = size(prob)

    # Objective
    obj = prob.obj

    # Extract variables
    Z = prob.Z; K = solver.K; d = solver.d;
    S = solver.S
    Q = solver.Q

    # Terminal cost-to-go
    S.xx[N] = Q.xx[N]
    S.x[N] = Q.x[N]

    # Initialize expecte change in cost-to-go
    ΔV = @SVector zeros(2)

    k = N-1
    while k > 0
        ix = Z[k]._x
        iu = Z[k]._u

        fdx = solver.∇F[k][ix,ix]
        fdu = solver.∇F[k][ix,iu]

        Qk = getQ(Q,k)

        Q.x[k] += fdx'S.x[k+1]
        Q.u[k] += fdu'S.x[k+1]
        Q.xx[k] += fdx'S.xx[k+1]*fdx
        Q.uu[k] += fdu'S.xx[k+1]*fdu
        Q.ux[k] += fdu'S.xx[k+1]*fdx

        if solver.opts.bp_reg_type == :state
            Quu_reg = Q.uu[k] + solver.ρ[1]*fdu'fdu
            Qux_reg = Q.ux[k] + solver.ρ[1]*fdu'fdx
        elseif solver.opts.bp_reg_type == :control
            Quu_reg = Q.uu[k] + solver.ρ[1]*I
            Qux_reg = Q.ux[k]
        end

        # Regularization

        # Compute gains
        K[k] = -(Quu_reg\Qux_reg)
        d[k] = -(Quu_reg\Q.u[k])

        # Calculate cost-to-go (using unregularized Quu and Qux)
        S.x[k]  =  Q.x[k] + K[k]'*Q.uu[k]*d[k] + K[k]'* Q.u[k] + Q.ux[k]'d[k]
        S.xx[k] = Q.xx[k] + K[k]'*Q.uu[k]*K[k] + K[k]'*Q.ux[k] + Q.ux[k]'K[k]
        S.xx[k] = 0.5*(S.xx[k] + S.xx[k]')

        # calculated change is cost-to-go over entire trajectory
        ΔV += @SVector [d[k]'*Q.u[k], 0.5*d[k]'*Q.uu[k]*d[k]]

        k -= 1
    end

    return ΔV
    
end

function _ctg_expansion(Q, S, A, B)
    (x=Q.x + A'S.x,
     u=Q.u + B'S.x,
     xx=Q.xx + A'S.xx*A,
     uu=Q.uu + B'S.xx*B,
     ux=Q.ux + B'S.xx*A)
end



function _cost_to_go(Q,K,d)
    ΔV = @SVector [d'Q.u, 0.5*d'Q.uu*d]
    Sx = Q.x + K'Q.uu*d + K'Q.u + Q.ux'd
    Sxx = Q.xx + K'Q.uu*K + K'Q.ux + Q.ux'K
    return ΔV, Sx, Sxx
end



function backwardpass!(prob::Problem, solver::StaticiLQRSolver)
    n,m,N = size(prob)

    # Objective
    obj = prob.obj

    X = prob.X; U = prob.U; K = solver.K; d = solver.d

    S = solver.S
    Q = solver.Q

    # Terminal cost-to-go
    S[N].xx = copy(Q[N].xx)
    S[N].x = copy(Q[N].x)

    # Initialize expected change in cost-to-go
    ΔV = zeros(2)

    # ix = @SVector [i for i = 1:n]
    # iu = @SVector [i for i = 1:m]
    ix, iu = 1:n, n .+ (1:m)

    # Backward pass
    k = N-1
    while k >= 1
        fdx, fdu = solver.∇F[k][ix,ix], solver.∇F[k][ix,iu]

        Q[k].x += fdx'*S[k+1].x
        Q[k].u += fdu'*S[k+1].x
        Q[k].xx += fdx'*S[k+1].xx*fdx
        Q[k].uu += fdu'*S[k+1].xx*fdu
        Q[k].ux += fdu'*S[k+1].xx*fdx

        if solver.opts.bp_reg_type == :state
            # Quu_reg = cholesky(Q[k].uu + solver.ρ[1]*fdu'*fdu,check=false)
            Quu_reg = Q[k].uu + solver.ρ[1]*fdu'*fdu
            Qux_reg = Q[k].ux + solver.ρ[1]*fdu'*fdx
        elseif solver.opts.bp_reg_type == :control
            # Quu_reg = cholesky(Q[k].uu + solver.ρ[1]*I,check=false)
            Quu_reg = Q[k].uu + solver.ρ[1]*Diagonal(ones(prob.model.m))
            Qux_reg = Q[k].ux
        end



        # Regularization
        # if Quu_reg.info == -1
        if !isposdef(Hermitian(Array(Quu_reg)))
            # increase regularization

            @logmsg InnerIters "Regularizing Quu "
            regularization_update!(solver,:increase)

            # reset backward pass
            k = N-1
            ΔV[1] = 0.
            ΔV[2] = 0.
            continue
        end

        # Compute gains
        K[k] = -1.0*(Quu_reg\Qux_reg)
        d[k] = -1.0*(Quu_reg\Q[k].u)

        # Calculate cost-to-go (using unregularized Quu and Qux)
        S[k].x  =  Q[k].x + K[k]'*Q[k].uu*d[k] + K[k]'* Q[k].u + Q[k].ux'*d[k]
        S[k].xx = Q[k].xx + K[k]'*Q[k].uu*K[k] + K[k]'*Q[k].ux + Q[k].ux'*K[k]
        S[k].xx = 0.5*(S[k].xx + S[k].xx')

        # calculated change is cost-to-go over entire trajectory
        ΔV[1] += d[k]'*Q[k].u
        ΔV[2] += 0.5*d[k]'*Q[k].uu*d[k]

        k = k - 1;
    end

    # decrease regularization after backward pass
    # regularization_update!(solver,:decrease)

    return ΔV
end

"""
$(SIGNATURES)
Propagate dynamics with a line search (in-place)
"""
function forwardpass!(prob::Problem, solver::StaticiLQRSolver, ΔV::Array, J_prev::Float64)
    # Pull out values from results
    X = prob.X; U = prob.U; X̄ = solver.X̄; Ū = solver.Ū

    J = Inf
    alpha = 1.0
    iter = 0
    z = -1.
    expected = 0.

    logger = current_logger()
    to = solver.stats[:timer]
    # @logmsg InnerIters :iter value=0
    # @logmsg InnerIters :cost value=J_prev
    while (z ≤ solver.opts.line_search_lower_bound || z > solver.opts.line_search_upper_bound) && J >= J_prev

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            # set trajectories to original trajectory
            copyto!(X̄,X)
            copyto!(Ū,U)

            J = cost(prob.obj, X̄, Ū, get_dt_traj(prob,Ū))

            z = 0.
            alpha = 0.0
            expected = 0.

            # @logmsg InnerLoop "Max iterations (forward pass)"
            regularization_update!(solver,:increase) # increase regularization
            solver.ρ[1] += solver.opts.bp_reg_fp
            break
        end

        # Otherwise, rollout a new trajectory for current alpha
        @timeit to "rollout" flag = rollout!(prob,solver,alpha)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            # @logmsg InnerIters "Non-finite values in rollout"
            iter += 1
            alpha /= 2.0
            continue
        end

        # Calcuate cost
        @timeit to "cost" J = cost(prob.obj, X̄, Ū, get_dt_traj(prob,Ū))   # Unconstrained cost

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
        # @logmsg InnerIters :iter value=iter
        # @logmsg InnerIters :α value=2*alpha
        # @logmsg InnerIters :cost value=J
        # @logmsg InnerIters :z value=z

    end  # forward pass loop

    # @logmsg InnerLoop :cost value=J
    # @logmsg InnerLoop :dJ value=J_prev-J
    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*alpha
    @logmsg InnerLoop :ρ value=solver.ρ[1]

    if J > J_prev
        error("Error: Cost increased during Forward Pass")
    end

    return J
end

"Simulate state trajectory with feedback control"
function rollout!(prob::StaticProblem{T}, solver::StaticiLQRSolver{T},alpha::T=1.0) where T
    Z = prob.Z
    K = solver.K; d = solver.d; X̄ = solver.X̄; Ū = solver.Ū
    X̄[1] = prob.x0
    Dt = get_dt_traj(prob)

    for k = 2:prob.N
        # Calculate state trajectory difference
        δx = state_diff(X̄[k-1],X[k-1],prob,solver)

        # Calculate updated control
        Ū[k-1] = U[k-1] + K[k-1]*δx + alpha*d[k-1]

        # Propagate dynamics
        X̄[k] = dynamics(prob.model, X̄[k-1], Ū[k-1], Dt[k])

        # Check that rollout has not diverged
        if ~(norm(X̄[k],Inf) < solver.opts.max_state_value && norm(Ū[k-1],Inf) < solver.opts.max_control_value)
            return false
        end
    end
    return true
end



function state_diff(x̄::AbstractVector{T}, x::AbstractVector{T}, prob::Problem{T},  solver::StaticiLQRSolver{T}) where T
    if true
        x̄ - x
    else
        nothing #TODO quaternion
    end
end
