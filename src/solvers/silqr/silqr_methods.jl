
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

function forwardpass!(prob::StaticProblem, solver::StaticiLQRSolver, ΔV, J_prev)
    Z = prob.Z; Z̄ = prob.Z̄
    obj = prob.obj

    J::Float64 = Inf 
    α = 1.0
    iter = 0
    z = -1.0
    expected = 0.0
    flag = true

    while (z ≤ solver.opts.line_search_lower_bound || z > solver.opts.line_search_upper_bound) && J >= J_prev

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            for k in eachindex(Z)
                Z̄[k] = Z[k]
            end
            J = cost(obj, Z̄)

            z = 0
            α = 0.0
            expected = 0.0

            regularization_update!(solver, :increase)
            solver.ρ[1] += solver.opts.bp_reg_fp
            break
        end

        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout!(prob, solver, α)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            # @logmsg InnerIters "Non-finite values in rollout"
            iter += 1
            α /= 2.0
            continue
        end


        # Calcuate cost
        J = cost(obj, Z̄)

        expected::Float64 = -α*(ΔV[1] + α*ΔV[2])
        if expected > 0.0
            z::Float64  = (J_prev - J)/expected
        else
            z = -1.0
        end

        iter += 1
        α /= 2.0
    end

    if J > J_prev
        # error("Error: Cost increased during Forward Pass")
    end

    return J

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

function rollout!(prob::StaticProblem, solver::StaticiLQRSolver, α=1.0)
    Z = prob.Z; Z̄ = prob.Z̄
    K = solver.K; d = solver.d;

    Z̄[1].z = [prob.x0; control(Z[1])]

    temp = 0.0

    for k = 1:prob.N-1
        δx = state(Z̄[k]) - state(Z[k])
        δu = K[k]*δx + α*d[k]
        Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]

        propagate_dynamics(prob.model, Z̄[k+1], Z̄[k])

        temp = norm(Z̄[k+1].z)
        if temp > solver.opts.max_state_value
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
