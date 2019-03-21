
function solve!(prob::Problem{T},solver::iLQRSolver{T}) where T
    n,m,N = size(prob)

    # Initial rollout
    rollout!(prob)

    for i = 1:solver.iterations
        step!(prob,solver)

        dJ = J - J_prev
        record_iteration!(prob,solver,dJ)
        evaluate_convergence(solver)
    end
end

function step!(prob::Problem, solver::iLQRSolver)
    jacobian!(prob,solver)

    ΔV = backwardpass!(prob,solver)
    J = forwardpass!(prob,solver,ΔV,J_prev)
end

function record_iteration!(prob::Problem,solver::iLQRSolver,dJ)
    solver.stats[:iterations] += 1
    push!(solver.stats[:dJ], dJ)
    push!(solver.stats[:gradient],calculate_gradient(prob,solver))
    dJ == 0 ? solver.stats[:dJ_zero_counter] += 1 : solver.stats[:dJ_zero_counter] = 0
end

function calculate_gradient(prob::Problem,solver::iLQRSolver)
    if solver.opts.gradient_type == :todorov
        gradient = gradient_todorov(prob,solver)
    elseif solver.opts.gradient_type == :feedforward
        gradient = gradient_feedforward(solver)
    end
    return gradient
end

"""
$(SIGNATURES)
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function gradient_todorov(prob::Problem,solver::iLQRSolver)
    N = prob.N
    maxes = zeros(N)
    for k = 1:N-1
        maxes[k] = maximum(abs.(solver.d[k])./(abs.(prob.U[k]).+1))
    end
    mean(maxes)
end

"""
$(SIGNATURES)
    Calculate the infinity norm of the gradient using feedforward term d (from δu = Kδx + d)
"""
function gradient_feedforward(solver::iLQRSolver)
    norm(solver.d,Inf)
end

function evaluate_convergence(solver::iLQRSolver)
    # Check for cost convergence
    # note the  dJ > 0 criteria exists to prevent loop exit when forward pass makes no improvement
    if 0.0 < dJ < solver.opts.cost_tolerance
        return true
    end

    # Check for gradient convergence
    gradient = solver.stats[:gradient][end]
    if gradient < solver.opts.gradient_norm_tolerance
        return true
    end

    # Check total iterations
    if solver.stats[:iterations] >= solver.opts.iterations
        return true
    end

    # Outer loop update if forward pass is repeatedly unsuccessful
    if solver.stats[:dJ_zero_counter] > solver.opts.dJ_counter_limit
        return true
    end
    return false
end


function regularization_update!(solver::iLQRSolver,status::Symbol=:increase)
    if status == :increase # increase regularization
        # @logmsg InnerLoop "Regularization Increased"
        solver.dρ[1] = max(solver.dρ[1]*solver.opts.bp_reg_increase_factor, solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = max(solver.ρ[1]*solver.dρ[1], solver.opts.bp_reg_min)
        if results.ρ[1] > solver.opts.bp_reg_max
            @warn "Max regularization exceeded"
        end
    elseif status == :decrease # decrease regularization
        results.dρ[1] = min(solver.dρ[1]/solver.opts.bp_reg_increase_factor, 1.0/solver.opts.bp_reg_increase_factor)
        results.ρ[1] = solver.ρ[1]*solver.dρ[1]*(solver.ρ[1]*solver.dρ[1]>solver.opts.bp_reg_min)
    end
end
