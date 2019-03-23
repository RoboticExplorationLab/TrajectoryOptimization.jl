"Augmented Lagrangian solve"
function solve!(prob::Problem, solver::AugmentedLagrangianSolver)
    unconstrained_solver = AbstractSolver(prob, solver.opts.unconstrained_solver)

    for i = 1:solver.opts.iterations
        J = step!(prob, solver, unconstrained_solver)
        c_max = max_violation(solver)
        record_iteration!(prob, solver, J, c_max, unconstrained_solver)
        evaluate_convergence(solver) ? break : nothing
    end
end

"Augmented Lagrangian step"
function step!(prob::Problem, solver::AugmentedLagrangianSolver,unconstrained_solver::AbstractSolver)
    J = solve!(prob, unconstrained_solver)
    dual_update!(prob, solver)
    penalty_update!(prob, solver)

    copyto!(solver.C_prev,solver.C)

    return J
end

function evaluate_convergence(solver::AugmentedLagrangianSolver)
    solver.stats[:c_max][end] < solver.opts.constraint_tolerance ? true : false
end

function record_iteration!(prob::Problem{T}, solver::AugmentedLagrangianSolver{T}, J::T, c_max::T, unconstrained_solver::AbstractSolver{T}) where T
    solver.stats[:iterations] += 1
    push!(solver.stats[:cost],J)
    push!(solver.stats[:c_max],c_max)
    push!(solver.stats_uncon, unconstrained_solver.stats)
end

"Saturate a vector element-wise with upper and lower bounds"
saturate(input::AbstractVector{T}, max_value::T, min_value::T) where T = max.(min_value, min.(max_value, input))

"Dual update (first-order)"
function dual_update!(prob::Problem, solver::AugmentedLagrangianSolver)
    c = solver.C; λ = solver.λ; μ = solver.μ

    for k = 1:prob.N
        copyto!(λ[k],saturate(λ[k] + μ[k].*c[k], solver.opts.dual_max,
            solver.opts.dual_min))
        copyto!(λ[k].inequality,max.(0.0, λ[k].inequality))
    end
end

"Penalty update (default) - update all penalty parameters"
function penalty_update!(prob::Problem, solver::AugmentedLagrangianSolver)
    μ = solver.μ
    for k = 1:prob.N
        copyto!(μ[k], saturate(solver.opts.penalty_scaling * μ[k], solver.opts.penalty_max, 0.0))
    end
end

"Generate augmented Lagrangian cost from unconstrained cost"
function AugmentedLagrangianCost(prob::Problem{T},solver::AugmentedLagrangianSolver{T}) where T
    AugmentedLagrangianCost{T}(prob.cost,prob.constraints,solver.C,solver.∇C,solver.λ,solver.μ,solver.active_set)
end

"Generate augmented Lagrangian problem from constrained problem"
function AugmentedLagrangianProblem(prob::Problem{T},solver::AugmentedLagrangianSolver{T}) where T
    al_cost = AugmentedLagrangianCost(prob,solver)
    al_prob = update_problem(prob,cost=al_cost,constraints=AbstractConstraint[],newProb=false)
end

"Evaluate maximum constraint violation"
function max_violation(solver::AugmentedLagrangianSolver)
    c_max = 0.0
    for k = 1:length(solver.C)
        c_max = max(norm(solver.C[k].equality,Inf),norm(solver.C[k].inequality .* solver.active_set[k].inequality,Inf), c_max)
    end
    return c_max
end
