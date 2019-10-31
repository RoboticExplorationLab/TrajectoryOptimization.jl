
# Generic solve methods
"iLQR solve method (non-allocating)"
function solve!(prob::StaticProblem, solver::StaticiLQRSolver{T}) where T<:AbstractFloat
    solver.stats.iterations = 0
    # reset!(solver)
    # to = solver.stats[:timer]
    Z = prob.Z; Z̄ = prob.Z̄;


    n,m,N = size(prob)
    J = Inf
    _J = get_J(prob.obj)

    # logger = default_logger(solver)

    # Initial rollout
    rollout!(prob)
    # live_plotting(prob,solver)

    cost!(prob.obj, prob.Z)
    J_prev = sum(_J)

    for i = 1:solver.opts.iterations
        J = step!(prob, solver, J_prev)

        # check for cost blow up
        if J > solver.opts.max_cost_value
            # @warn "Cost exceeded maximum cost"
            return solver
        end

        for k = 1:N
            Z[k].z = Z̄[k].z
        end

        dJ = abs(J - J_prev)
        J_prev = copy(J)
        gradient_todorov!(prob, solver)

        record_iteration!(solver, J, dJ)
        evaluate_convergence(solver) ? break : nothing
    end
    return solver
end

"""
Take one step of iLQR algorithm (non-allocating)
"""
function step!(prob::StaticProblem, solver::StaticiLQRSolver, J)
    discrete_jacobian!(solver.∇F, prob.model, prob.Z)
    cost_expansion(solver.Q, prob.obj, prob.Z)
    ΔV = backwardpass!(prob, solver)
    forwardpass!(prob, solver, ΔV, J)
end

"""
Stash iteration statistics
"""
function record_iteration!(solver::StaticiLQRSolver, J, dJ)
    solver.stats.iterations += 1
    i = solver.stats.iterations::Int
    solver.stats.cost[i] = J
    solver.stats.dJ[i] = dJ
    solver.stats.gradient[i] = mean(solver.grad)
    return nothing
end

"""
$(SIGNATURES)
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function gradient_todorov!(prob::StaticProblem, solver::StaticiLQRSolver)
    for k in eachindex(solver.d)
        solver.grad[k] = maximum( abs.(solver.d[k]) ./ (abs.(control(prob.Z[k])) .+ 1) )
    end
end

"""
$(SIGNATURES)
Check convergence conditions for iLQR
"""
function evaluate_convergence(solver::StaticiLQRSolver)
    # Get current iterations
    i = solver.stats.iterations

    # Check for cost convergence
    # note the  dJ > 0 criteria exists to prevent loop exit when forward pass makes no improvement
    if 0.0 < solver.stats.dJ[i] < solver.opts.cost_tolerance
        return true
    end

    # Check for gradient convergence
    if solver.stats.gradient[i] < solver.opts.gradient_norm_tolerance
        return true
    end

    # Check total iterations
    if i >= solver.opts.iterations
        return true
    end

    # Outer loop update if forward pass is repeatedly unsuccessful
    if solver.stats.dJ_zero_counter > solver.opts.dJ_counter_limit
        return true
    end

    return false
end

"""
$(SIGNATURES)
Calculates the optimal feedback gains K,d as well as the 2nd Order approximation of the
Cost-to-Go, using a backward Riccati-style recursion. (non-allocating)
"""
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

    regularization_update!(solver, :decrease)

    return ΔV

end

"""
$(SIGNATURES)
Simulate the system forward using the optimal feedback gains from the backward pass,
projecting the system on the dynamically feasible subspace. Performs a line search to ensure
adequate progress on the nonlinear problem.
"""
function forwardpass!(prob::StaticProblem, solver::StaticiLQRSolver, ΔV, J_prev)
    Z = prob.Z; Z̄ = prob.Z̄
    obj = prob.obj

    _J = get_J(prob.obj)
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
            cost!(obj, Z̄)
            J = sum(_J)

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
        cost!(obj, Z̄)
        J = sum(_J)

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
        error("Error: Cost increased during Forward Pass")
    end

    return J

end


"""
$(SIGNATURES)
Simulate forward the system with the optimal feedback gains from the iLQR backward pass.
(non-allocating)
"""
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

"Simulate the forward the dynamics open-loop"
function rollout!(prob::StaticProblem)
    prob.Z[1].z = [prob.x0; control(prob.Z[1])]
    for k = 2:prob.N
        propagate_dynamics(prob.model, prob.Z[k], prob.Z[k-1])
    end
end

"""
$(SIGNATURES)
Update the regularzation for the iLQR backward pass
"""
function regularization_update!(solver::StaticiLQRSolver,status::Symbol=:increase)
    # println("reg $(status)")
    if status == :increase # increase regularization
        # @logmsg InnerLoop "Regularization Increased"
        solver.dρ[1] = max(solver.dρ[1]*solver.opts.bp_reg_increase_factor, solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = max(solver.ρ[1]*solver.dρ[1], solver.opts.bp_reg_min)
        # if solver.ρ[1] > solver.opts.bp_reg_max
        #     @warn "Max regularization exceeded"
        # end
    elseif status == :decrease # decrease regularization
        # TODO: Avoid divides by storing the decrease factor (divides are 10x slower)
        solver.dρ[1] = min(solver.dρ[1]/solver.opts.bp_reg_increase_factor, 1.0/solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = solver.ρ[1]*solver.dρ[1]*(solver.ρ[1]*solver.dρ[1]>solver.opts.bp_reg_min)
    end
end
