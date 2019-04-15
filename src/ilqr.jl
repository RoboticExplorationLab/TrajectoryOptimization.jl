function solve!(prob::Problem{T},opts::AbstractSolverOptions{T}) where T
    solver = AbstractSolver(prob,opts)
    solve!(prob,solver)
end

function solve(prob0::Problem{T},solver::AbstractSolver)::Problem{T} where T
    prob = copy(prob0)
    solve!(prob,solver)
    return prob
end

function solve(prob0::Problem{T},opts::AbstractSolverOptions{T})::Problem{T} where T
    prob = copy(prob0)
    solver = AbstractSolver(prob,opts)
    solve!(prob,solver)
    return prob
end

function solve!(prob::Problem{T}, solver::L1Solver{T}) where T
    reset!(solver)
    println("***We are in solve!(prob::Problem{T}, solver::L1Solver{T})")
    n,m,N = size(prob)
    J = Inf
    # # Create Trajectories
    # C          = [BlockArray(zeros(T,p),c_part)       for k = 1:N-1]
    # ∇C         = [BlockArray(zeros(T,p,n+m),c_part2)  for k = 1:N-1]
    # λ          = [BlockArray(ones(T,p),c_part) for k = 1:N-1]
    # μ          = [BlockArray(ones(T,p),c_part) for k = 1:N-1]
    # active_set = [BlockArray(ones(Bool,p),c_part)     for k = 1:N-1]
    # push!(C,BlockVector(T,c_term))
    # push!(∇C,BlockMatrix(T,c_term,n,0))
    # push!(λ,BlockVector(T,c_term))
    # push!(μ,BlockArray(ones(T,num_constraints(c_term)), create_partition(c_term)))
    # push!(active_set,BlockVector(Bool,c_term))

    # Initialization
    Y = zeros(N,m) ###
    M = ones(N,m) ###
    ρ = 10^-4 ###
    logger = default_logger(solver)

    # Initial rollout
    rollout!(prob)
    J_prev = cost(prob.cost, prob.X, prob.U, prob.dt)
    push!(solver.stats[:cost], J_prev)

    # create L1ALCost
    l1cost = prob.cost
    l1alcost = L1ALCost(l1cost,Y,M,ρ)
    # create L1ALProblem
    l1alprob = Problem(prob.model, l1alcost, prob.constraints, prob.x0, prob.X,
        prob.U, prob.N, prob.dt, prob.tf)
    # create L1ALSolver
    l1alsolver = iLQRSolver(l1alprob)

    with_logger(logger) do
        for i = 1:solver.opts.iterations
            # update L1AlCost, L1ALProblem, L1ALSolver
            l1alcost.Y = Y
            l1alcost.M = M
            l1alcost.ρ = ρ
            l1alprob.cost = l1alcost
            l1alsolver.prob = l1alproblem
            # ADMM Steps
            # Optimal Control Update
            X,U = solve(l1alprob,l1alsolver,U) ###
            # Soft threshold Update
            Y = soft_thresholding(cost.d/ρ, U + (M .- [cost.r])/ρ)
            # Dual Update
            M += ρ*(U - Y)

            # eval convergence
            J = cost(prob.cost, X, U, prob.dt)
            # check for cost blow up
            if J > solver.opts.max_cost_value
                error("Cost exceeded maximum cost")
            end
            dJ = abs(J - J_prev)
            J_prev = copy(J)
            record_iteration!(prob, solver, J, dJ)
            live_plotting(prob,solver)
            # println(logger, InnerLoop)
            evaluate_convergence(l1solver) ? break : nothing
        end
    end
    return J
end


function live_plotting(prob::Problem{T},solver::L1Solver{T}) where T
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

function record_iteration!(prob::Problem{T}, solver::L1Solver{T}, J::T, dJ::T) where T
    solver.stats[:iterations] += 1
    push!(solver.stats[:cost], J)
    push!(solver.stats[:dJ], dJ)
    push!(solver.stats[:gradient],calculate_gradient(prob,solver))
    dJ == 0 ? solver.stats[:dJ_zero_counter] += 1 : solver.stats[:dJ_zero_counter] = 0

    @logmsg InnerLoop :iter value=solver.stats[:iterations]
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ   value=dJ
    @logmsg InnerLoop :grad value=solver.stats[:gradient][end]
    @logmsg InnerLoop :zero_count value=solver.stats[:dJ_zero_counter][end]
end

function calculate_gradient(prob::Problem,solver::L1Solver)
    if solver.opts.gradient_type == :todorov
        gradient = gradient_todorov(prob,solver)
    elseif solver.opts.gradient_type == :feedforward
        gradient = gradient_feedforward(solver)
    end
    return gradient
end

function evaluate_convergence(solver::L1Solver)
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








function solve!(prob::Problem{T}, solver::iLQRSolver{T}) where T
    reset!(solver)

    n,m,N = size(prob)
    J = Inf

    logger = default_logger(solver)

    # Initial rollout
    rollout!(prob)
    J_prev = cost(prob.cost, prob.X, prob.U, prob.dt)
    push!(solver.stats[:cost], J_prev)

    with_logger(logger) do
        for i = 1:solver.opts.iterations
            J = step!(prob, solver, J_prev)

            # check for cost blow up
            if J > solver.opts.max_cost_value
                error("Cost exceeded maximum cost")
            end

            copyto!(prob.X, solver.X̄)
            copyto!(prob.U, solver.Ū)

            dJ = abs(J - J_prev)
            J_prev = copy(J)
            record_iteration!(prob, solver, J, dJ)
            live_plotting(prob,solver)

            println(logger, InnerLoop)
            evaluate_convergence(solver) ? break : nothing
        end
    end
    return J
end


function step!(prob::Problem{T}, solver::iLQRSolver{T}, J::T) where T
    jacobian!(prob,solver)
    ΔV = backwardpass!(prob,solver)
    forwardpass!(prob,solver,ΔV,J)
end

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
    dJ == 0 ? solver.stats[:dJ_zero_counter] += 1 : solver.stats[:dJ_zero_counter] = 0

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
