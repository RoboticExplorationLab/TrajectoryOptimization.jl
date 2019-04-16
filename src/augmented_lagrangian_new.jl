"Augmented Lagrangian solve"
function solve!(prob::Problem{T}, solver::AugmentedLagrangianSolver{T}) where T
    reset!(solver)

    solver_uncon = AbstractSolver(prob, solver.opts.opts_uncon)

    prob_al = AugmentedLagrangianProblem(prob, solver)
    logger = default_logger(solver)

    with_logger(logger) do
        for i = 1:solver.opts.iterations
            set_intermediate_uncon_solver_tolerances!(solver,solver_uncon,i)
            J = step!(prob_al, solver, solver_uncon)

            record_iteration!(prob, solver, J, solver_uncon)
            println(logger,OuterLoop)
            evaluate_convergence(solver) ? break : nothing
        end
    end
end

function solve!(prob::Problem{T},opts::AugmentedLagrangianSolverOptions{T}) where T
    isempty(prob.constraints) ? solver = AbstractSolver(prob,opts.opts_uncon) : solver = AbstractSolver(prob,opts)
    solve!(prob,solver)
end

function set_intermediate_uncon_solver_tolerances!(solver::AugmentedLagrangianSolver{T},
        solver_uncon::AbstractSolver{T},i::Int) where T
    if i != solver.opts.iterations
        solver_uncon.opts.cost_tolerance = solver.opts.cost_tolerance_intermediate
        solver_uncon.opts.gradient_norm_tolerance = solver.opts.gradient_norm_tolerance_intermediate
    else
        solver_uncon.opts.cost_tolerance = solver.opts.cost_tolerance
        solver_uncon.opts.gradient_norm_tolerance = solver.opts.gradient_norm_tolerance
    end

    return nothing
end

"Augmented Lagrangian step"
function step!(prob::Problem{T}, solver::AugmentedLagrangianSolver{T},
        unconstrained_solver::AbstractSolver) where T

    # Solve the unconstrained problem
    J = solve!(prob, unconstrained_solver)

    reset!(unconstrained_solver)

    # Outer loop update
    dual_update!(prob, solver)
    penalty_update!(prob, solver)
    copyto!(solver.C_prev,solver.C)

    return J
end

function evaluate_convergence(solver::AugmentedLagrangianSolver{T}) where T
    solver.stats[:c_max][end] < solver.opts.constraint_tolerance ? true : false
end

function record_iteration!(prob::Problem{T}, solver::AugmentedLagrangianSolver{T}, J::T,
        unconstrained_solver::AbstractSolver) where T
    c_max = max_violation(solver)

    solver.stats[:iterations] += 1
    solver.stats[:iterations_total] += unconstrained_solver.stats[:iterations]
    push!(solver.stats[:iterations_inner], unconstrained_solver.stats[:iterations])
    push!(solver.stats[:cost],J)
    push!(solver.stats[:c_max],c_max)
    push!(solver.stats_uncon, unconstrained_solver.stats)

    @logmsg OuterLoop :iter value=solver.stats[:iterations]
    @logmsg OuterLoop :total value=solver.stats[:iterations_total]
    @logmsg OuterLoop :cost value=J
    @logmsg OuterLoop :c_max value=c_max
end

"Saturate a vector element-wise with upper and lower bounds"
saturate(input::AbstractVector{T}, max_value::T, min_value::T) where T = max.(min_value, min.(max_value, input))

"Dual update (first-order)"
function dual_update!(prob::Problem{T}, solver::AugmentedLagrangianSolver{T}) where T
    c = solver.C; λ = solver.λ; μ = solver.μ

    for k = 1:prob.N
        copyto!(λ[k],saturate(λ[k] + μ[k].*c[k], solver.opts.dual_max,
            solver.opts.dual_min))
        copyto!(λ[k].inequality,max.(0.0, λ[k].inequality))
    end

    # Update active set after updating multipliers (need to calculate c_max)
    update_active_set!(solver.active_set, solver.C, solver.λ)
end

"Penalty update (default) - update all penalty parameters"
function penalty_update!(prob::Problem{T}, solver::AugmentedLagrangianSolver{T}) where T
    μ = solver.μ
    for k = 1:prob.N
        copyto!(μ[k], saturate(solver.opts.penalty_scaling * μ[k], solver.opts.penalty_max, 0.0))
    end
end

"Generate augmented Lagrangian cost from unconstrained cost"
function ALCost(prob::Problem{T},
        solver::AugmentedLagrangianSolver{T}) where T
    ALCost{T}(prob.cost,prob.constraints,solver.C,solver.∇C,solver.λ,solver.μ,solver.active_set)
end

"Generate augmented Lagrangian problem from constrained problem"
function AugmentedLagrangianProblem(prob::Problem{T},solver::AugmentedLagrangianSolver{T}) where T
    al_cost = ALCost(prob,solver)
    al_prob = update_problem(prob,cost=al_cost,constraints=AbstractConstraint[],newProb=false)
end

"Evaluate maximum constraint violation"
function max_violation(solver::AugmentedLagrangianSolver{T}) where T
    c_max = 0.0
    C = solver.C
    N = length(C)
    if length(C[1]) > 0
        for k = 1:N-1
            c_max = max(norm(C[k].equality,Inf), c_max)
            if length(C[k].inequality) > 0
                c_max = max(pos(maximum(C[k].inequality)), c_max)
            end
        end
    end
    if length(solver.C[N]) > 0
        c_max = max(norm(C[N].equality,Inf), c_max)
        if length(C[N].inequality) > 0
            c_max = max(pos(maximum(C[N].inequality)), c_max)
        end
    end
    return c_max
end
