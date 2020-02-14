
function solve!(solver::AugmentedLagrangianSolver{T,S}) where {T,S}
	initialize!(solver)
    c_max::T = Inf

	conSet = get_constraints(solver)
	solver_uncon = solver.solver_uncon::S

	# Calculate cost
    J_ = get_J(get_objective(solver))
    J = sum(J_)

    for i = 1:solver.opts.iterations
		set_tolerances!(solver, solver_uncon, i)

        step!(solver)
        J = sum(J_)
        c_max = maximum(conSet.c_max)

        record_iteration!(solver, J, c_max)

        converged = evaluate_convergence(solver)
        if converged
            break
        end

        reset!(solver_uncon, false)
    end
    return solver
end

function initialize!(solver::AugmentedLagrangianSolver)
	set_verbosity!(solver.opts)
	clear_cache!(solver.opts)

	# Reset solver
    reset!(get_constraints(solver), solver.opts)
    solver.stats.iterations = 0
	solver.stats.iterations_total = 0

	# Calculate cost
    cost!(get_objective(solver), get_trajectory(solver))
end

function step!(solver::AugmentedLagrangianSolver)

    # Solve the unconstrained problem
    solve!(solver.solver_uncon)

    # Outer loop update
    dual_update!(solver)
    penalty_update!(solver)
    max_violation!(get_constraints(solver))

	# Reset verbosity level after it's modified
	set_verbosity!(solver.opts)
end

function record_iteration!(solver::AugmentedLagrangianSolver{T,S}, J::T, c_max::T) where {T,S}

    solver.stats.iterations += 1
    i = solver.stats.iterations
    solver.stats.iterations_total += solver.solver_uncon.stats.iterations
    solver.stats.c_max[i] = c_max
	solver.stats.cost[i] = J

	conSet = get_constraints(solver)
	max_penalty!(conSet)
	solver.stats.penalty_max[i] = maximum(conSet.c_max)

	@logmsg OuterLoop :iter value=i
	@logmsg OuterLoop :total value=solver.stats.iterations_total
	@logmsg OuterLoop :cost value=J
    @logmsg OuterLoop :c_max value=c_max
	if solver.opts.verbose
		print_level(OuterLoop)
	end
end

function set_tolerances!(solver::AugmentedLagrangianSolver{T},
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

function evaluate_convergence(solver::AugmentedLagrangianSolver)
	i = solver.stats.iterations
    solver.stats.c_max[i] < solver.opts.constraint_tolerance ||
		solver.stats.penalty_max[i] >= solver.opts.penalty_max
end

"General Dual Update"
function dual_update!(solver::AugmentedLagrangianSolver) where {T,Q,N,M,NM}
    conSet = get_constraints(solver)
    for i in eachindex(conSet.constraints)
        dual_update!(conSet.constraints[i])
    end
end

"Dual Update for Equality Constraints"
function dual_update!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractConstraint{Equality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	λ_max = con.params.λ_max
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], -λ_max, λ_max)
	end
end

"Dual Update for Inequality Constraints"
function dual_update!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractConstraint{Inequality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], 0.0, con.params.λ_max)
	end
end

"General Penalty Update"
function penalty_update!(solver::AugmentedLagrangianSolver)
    conSet = get_constraints(solver)
    for i in eachindex(conSet.constraints)
        penalty_update!(conSet.constraints[i])
    end
end

"Penalty Update for ConstraintVals"
function penalty_update!(con::ConstraintVals{T}) where T
	ϕ = con.params.ϕ
	μ = con.μ
	for i in eachindex(con.inds)
		μ[i] = clamp.(ϕ * μ[i], 0.0, con.params.μ_max)
	end
end
