"Augmented Lagrangian solve"
function solve!(prob::Problem, solver::AugmentedLagrangianSolver)
    unconstrained_solver = AbstractSolver(prob, solver.opts.unconstrained_solver)

    for i = 1:solver.opts.iterations
        J = step!(prob, solver, unconstrained_solver)
        c_max = max_violation(solver)
        record_iteration(prob, solver, J, c_max)
        evaluate_convergence(solver) ? break : nothing
    end

    solver.stats_uncon = unconstrained_solver.stats
end

"Augmented Lagrangian step"
function step!(prob::Problem, solver::AugmentedLagrangianSolver,
        unconstrained_solver::AbstractSolver)
    J = solve!(prob, unconstrained_solver)
    dual_update!(prob, solver)
    penalty_update!(prob, solver)

    solver.C_prev .= deepcopy(solver.C)

    return J
end

function evaluate_convergence(solver::AugmentedLagrangianSolver)
    solver.stats[:c_max][end] < solver.opts.constraint_tolerance ? true : false
end

function record_iteration!(prob::Problem{T}, solver::AugmentedLagrangianSolver{T},
        J::T, c_max::T) where T
    solver.stats[:iterations] += 1
    push!(solver.stats[:cost],J)
    push!(solver.stats[:c_max],c_max)
end

"Saturate a vector element-wise with upper and lower bounds"
saturate(input::AbstractVector{T}, max_value::T, min_value::T) where T = max.(min_value, min.(max_value, input))

"Dual update (first-order)"
function dual_update!(prob::Problem, solver::AugmentedLagrangianSolver)
    c = solver.C; λ = solver.λ; μ = solver.μ

    for k = 1:prob.N
        λ[k] = saturate(λ[k] + μ[k].*c[k], solver.opts.dual_max,
            solver.opts.dual_min)
        λ[k].inequality = max.(0.0, λ[k].inequality)
    end
end

"Penalty update (default) - update all penalty parameters"
function penalty_update!(prob::Problem, solver::AugmentedLagrangianSolver)
    μ = solver.μ
    for k = 1:prob.N
        μ[k] = saturate(solver.opts.penalty_scaling * μ[k], solver.opts.penalty_max, 0.0)
    end
end
