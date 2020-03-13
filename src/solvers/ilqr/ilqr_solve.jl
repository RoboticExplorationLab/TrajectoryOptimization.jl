function initialize!(solver::iLQRSolver2)
    set_verbosity!(solver.opts)
    clear_cache!(solver.opts)

    solver.ρ[1] = solver.opts.bp_reg_initial
    solver.dρ[1] = 0.0

    # Initial rollout
    rollout!(solver)
    cost!(solver.obj, solver.Z)
end

# Generic solve methods
"iLQR solve method (non-allocating)"
function solve!(solver::iLQRSolver{T}) where T<:AbstractFloat
	initialize!(solver)

    solver.stats.iterations = 0
    Z = solver.Z; Z̄ = solver.Z̄;

    n,m,N = size(solver)
    J = Inf
    _J = get_J(solver.obj)
    J_prev = sum(_J)

    for i = 1:solver.opts.iterations
        J = step!(solver, J_prev)

        # check for cost blow up
        if J > solver.opts.max_cost_value
            # @warn "Cost exceeded maximum cost"
            return solver
        end

        copy_trajectories!(solver)

        dJ = abs(J - J_prev)
        J_prev = copy(J)
        gradient_todorov!(solver)

        record_iteration!(solver, J, dJ)
        evaluate_convergence(solver) ? break : nothing
    end
    return solver
end

function step!(solver::iLQRSolver2, J)
    Z = solver.Z
    RobotDynamics.state_diff_jacobian!(solver.G, solver.model, Z)
	RobotDynamics.dynamics_expansion!(solver.D, solver.model, solver.Z)
    cost_expansion!(solver.Q, solver.obj, solver.Z)
	error_expansion!(solver.D, solver.model, solver.G)
	error_expansion!(solver.Q, solver.model, Z, solver.G)
	if solver.opts.static_bp
    	ΔV = static_backwardpass!(solver)
	else
		ΔV = backwardpass!(solver)
	end
    forwardpass!(solver, ΔV, J)
end

"""
$(SIGNATURES)
Simulate the system forward using the optimal feedback gains from the backward pass,
projecting the system on the dynamically feasible subspace. Performs a line search to ensure
adequate progress on the nonlinear problem.
"""
function forwardpass!(solver::iLQRSolver, ΔV, J_prev)
    Z = solver.Z; Z̄ = solver.Z̄
    obj = solver.obj

    _J = get_J(obj)
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
                Z̄[k].z = Z[k].z
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
        flag = rollout!(solver, α)

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

    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*α
    @logmsg InnerLoop :ρ value=solver.ρ[1]

    return J

end

function copy_trajectories!(solver::iLQRSolver)
    for k = 1:solver.N
        solver.Z[k].z = solver.Z̄[k].z
    end
end

"""
Stash iteration statistics
"""
function record_iteration!(solver::iLQRSolver, J, dJ)
    solver.stats.iterations += 1
    i = solver.stats.iterations::Int
    solver.stats.cost[i] = J
    solver.stats.dJ[i] = dJ
    solver.stats.gradient[i] = mean(solver.grad)

    @logmsg InnerLoop :iter value=i
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ   value=dJ
    @logmsg InnerLoop :grad value=solver.stats.gradient[i]
    # @logmsg InnerLoop :zero_count value=solver.stats[:dJ_zero_counter][end]
    if solver.opts.verbose
        print_level(InnerLoop)
    end
    return nothing
end

"""
$(SIGNATURES)
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function gradient_todorov!(solver::iLQRSolver2)
	tmp = solver.S[end].u
    for k in eachindex(solver.d)
		tmp .= abs.(solver.d[k])
		u = abs.(control(solver.Z[k])) .+ 1
		tmp ./= u
		solver.grad[k] = maximum(tmp)
    end
end


"""
$(SIGNATURES)
Check convergence conditions for iLQR
"""
function evaluate_convergence(solver::iLQRSolver)
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
Update the regularzation for the iLQR backward pass
"""
function regularization_update!(solver::iLQRSolver,status::Symbol=:increase)
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
