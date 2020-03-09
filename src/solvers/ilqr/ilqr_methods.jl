# Generic solve methods
"iLQR solve method"
function solve!(prob::Problem{T,Discrete}, solver::iLQRSolver{T}) where T<:AbstractFloat
    reset!(solver)
    to = solver.stats[:timer]

    n,m,N = size(prob)
    J = Inf

    logger = default_logger(solver)

    # Initial rollout
    rollout!(prob)
    live_plotting(prob,solver)

    J_prev = cost(prob.obj, prob.X, prob.U, get_dt_traj(prob))


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

function step!(prob::Problem{T}, solver::iLQRSolver{T}, J::T) where T
    to = solver.stats[:timer]
    @timeit to "jacobians" jacobian!(prob,solver)
    @timeit to "cost expansion" cost_expansion!(prob,solver)
    @timeit to "backward pass" ΔV = backwardpass!(prob,solver)
    @timeit to "forward pass" forwardpass!(prob,solver,ΔV,J)
end

"Plot state, control trajectories"
function live_plotting(prob::Problem{T},solver::iLQRSolver{T}) where T
    if solver.opts.live_plotting == :state
        p = plot(prob.X,title="State trajectory")
        display(p)
    elseif solver.opts.live_plotting == :control
        p = plot(prob.U,title="Control trajectory")
        display(p)
    else
        nothing
    end
end

function record_iteration!(prob::Problem{T}, solver::iLQRSolver{T}, J::T, dJ::T) where T
    solver.stats[:iterations] += 1
    push!(solver.stats[:cost], J)
    push!(solver.stats[:dJ], dJ)
    push!(solver.stats[:gradient],calculate_gradient(prob,solver))
    dJ == 0.0 ? solver.stats[:dJ_zero_counter] += 1 : solver.stats[:dJ_zero_counter] = 0

    @logmsg InnerLoop :iter value=solver.stats[:iterations]
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ   value=dJ
    @logmsg InnerLoop :grad value=solver.stats[:gradient][end]
    @logmsg InnerLoop :zero_count value=solver.stats[:dJ_zero_counter][end]
end

function calculate_gradient(prob::Problem,solver::iLQRSolver)
    if solver.opts.gradient_type == :todorov
        gradient = gradient_todorov(prob,solver)
    elseif solver.opts.gradient_type == :feedforward
        gradient = gradient_feedforward(solver)
    elseif solver.opts.gradient_type == :ℓ2
        gradient = norm(compute_gradient(prob,solver))
    elseif solver.opts.gradient_type == :ℓinf
        gradient = norm(compute_gradient(prob,solver),Inf)
    end
    return gradient
end

function compute_gradient(prob::Problem,solver::iLQRSolver)
    N = prob.N
    reset!(solver.Q)
    cost_expansion!(solver.Q,prob.obj,prob.X,prob.U,get_dt_traj(prob))
    ∇ℓ = []
    for k = 1:N-1
        push!(∇ℓ,solver.Q[k].x)
        push!(∇ℓ,solver.Q[k].u)
    end
    push!(∇ℓ,solver.Q[N].x)

    vcat(∇ℓ...)
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
    if 0.0 < solver.stats[:dJ][end] < solver.opts.cost_tolerance
        return true
    end

    # Check for gradient convergence
    if solver.stats[:gradient][end] < solver.opts.gradient_norm_tolerance
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
        if solver.ρ[1] > solver.opts.bp_reg_max
            @warn "Max regularization exceeded"
        end
    elseif status == :decrease # decrease regularization
        solver.dρ[1] = min(solver.dρ[1]/solver.opts.bp_reg_increase_factor, 1.0/solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = solver.ρ[1]*solver.dρ[1]*(solver.ρ[1]*solver.dρ[1]>solver.opts.bp_reg_min)
    end
end

"Project dynamically infeasible state trajectory into feasible space using TVLQR"
function projection!(prob::Problem{T},opts::iLQRSolverOptions{T}=iLQRSolverOptions{T}()) where T
    # backward pass - project infeasible trajectory into feasible space using time varying lqr
    solver_ilqr = AbstractSolver(prob, opts)
    backwardpass!(prob, solver_ilqr)

    # rollout
    rollout!(prob,solver_ilqr,0.0)

    # update trajectories
    copyto!(prob.X, solver_ilqr.X̄)
    copyto!(prob.U, solver_ilqr.Ū)
end
