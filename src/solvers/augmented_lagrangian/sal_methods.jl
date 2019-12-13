
function solve!(solver::StaticALSolver{T,S}) where {T,S}
    c_max::T = Inf

	# Extract stuff from solver
	Z = get_trajectory(solver)
	obj = get_objective(solver.solver_uncon)::StaticALObjective{T}
    conSet = obj.constraints
    solver_uncon = solver.solver_uncon::S

	# Reset solver
    reset!(conSet)
    solver.stats.iterations = 0

	# Calculate cost
    cost!(obj, Z)
    J_ = get_J(obj)
    J = sum(J_)


    for i = 1:solver.opts.iterations
        set_tolerances!(solver,solver_uncon,i)

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

function step!(solver::StaticALSolver)

    # Solve the unconstrained problem
    solve!(solver.solver_uncon)

    # Outer loop update
    dual_update!(solver)
    penalty_update!(solver)
    max_violation!(get_constraints(solver))

end

function record_iteration!(solver::StaticALSolver{T,S}, J::T, c_max::T) where {T,S}

    solver.stats.iterations += 1
    i = solver.stats.iterations
    solver.stats.iterations_total += solver.solver_uncon.stats.iterations
    solver.stats.c_max[i] = c_max
	solver.stats.cost[i] = J

	conSet = get_constraints(solver)
	max_penalty!(conSet)
	solver.stats.penalty_max[i] = maximum(conSet.c_max)
    # solver.stats.
    # push!(solver.stats[:iterations_inner], unconstrained_solver.stats[:iterations])
    # push!(solver.stats[:cost],J)
    # push!(solver.stats[:c_max],c_max)
    # push!(solver.stats[:penalty_max],max_penalty(solver))
    # push!(solver.stats_uncon, copy(unconstrained_solver.stats))
    #
end

function set_tolerances!(solver::StaticALSolver{T},
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

function evaluate_convergence(solver::StaticALSolver)
	i = solver.stats.iterations
    solver.stats.c_max[i] < solver.opts.constraint_tolerance ||
		solver.stats.penalty_max[i] >= solver.opts.penalty_max
end

"General Dual Update"
function dual_update!(solver::StaticALSolver) where {T,Q,N,M,NM}
    conSet = get_constraints(solver)
    for i in eachindex(conSet.constraints)
        dual_update!(conSet.constraints[i])
    end
end

"Dual Update for Equality Constraints"
function dual_update!(con::ConstraintVals{T,W,C}) where
		{T,W,C<:AbstractStaticConstraint{Equality}}
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
		{T,W,C<:AbstractStaticConstraint{Inequality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], 0.0, con.params.λ_max)
	end
end

"General Penalty Update"
function penalty_update!(solver::StaticALSolver)
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
