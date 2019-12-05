
function solve!(prob::StaticALProblem, solver::StaticALSolver{T,S}) where {T,S}
    c_max::T = Inf
    conSet = prob.obj.constraints
    reset!(conSet, solver.opts)
    solver.stats.iterations = 0
    solver_uncon = solver.solver_uncon::S
    cost!(prob.obj, prob.Z)
    J_ = get_J(prob.obj)
    J = sum(J_)


    for i = 1:solver.opts.iterations
        set_tolerances!(solver,solver_uncon,i)

        step!(prob, solver)
        J = sum(J_)
        c_max = maximum(conSet.c_max)


        record_iteration!(prob, solver, J, c_max)

        converged = evaluate_convergence(solver)
        if converged
            break
        end

        reset!(solver_uncon, false)
    end
    return solver
end

function step!(prob::StaticALProblem, solver::StaticALSolver)

    # Solve the unconstrained problem
    solve!(prob, solver.solver_uncon)

    # Outer loop update
    dual_update!(prob, solver)
    penalty_update!(prob, solver)
    max_violation!(prob.obj.constraints)
    # copyto!(solver.C_prev,solver.C)

end

function record_iteration!(prob::StaticProblem, solver::StaticALSolver{T,S},
        J::T, c_max::T) where {T,S}

    solver.stats.iterations += 1
    i = solver.stats.iterations
    solver.stats.iterations_total += solver.solver_uncon.stats.iterations
    solver.stats.c_max[i] = c_max
	solver.stats.cost[i] = J

	conSet = get_constraints(prob)
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
function dual_update!(prob::StaticALProblem,
        solver::StaticALSolver) where {T,Q,N,M,NM}
    conSet = prob.obj.constraints
    for i in eachindex(conSet.constraints)
        dual_update!(conSet.constraints[i], solver.opts)
    end
end

"Dual Update for Equality Constraints"
function dual_update!(con::ConstraintVals{T,W,C},
		opts::StaticALSolverOptions{T}) where
		{T,W,C<:AbstractStaticConstraint{Equality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], -opts.dual_max, opts.dual_max)
	end
end

"Dual Update for Inequality Constraints"
function dual_update!(con::ConstraintVals{T,W,C},
		opts::StaticALSolverOptions{T}) where
		{T,W,C<:AbstractStaticConstraint{Inequality}}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		λ[i] = clamp.(λ[i] + μ[i] .* c[i], 0.0, opts.dual_max)
	end
end

"General Penalty Update"
function penalty_update!(prob::StaticALProblem, solver::StaticALSolver)
    conSet = prob.obj.constraints
    for i in eachindex(conSet.constraints)
        penalty_update!(conSet.constraints[i], solver.opts)
    end
end

"Penalty Update for ConstraintVals"
function penalty_update!(con::ConstraintVals{T}, opts::StaticALSolverOptions{T}) where T
	ϕ = opts.penalty_scaling
	μ = con.μ
	for i in eachindex(con.inds)
		μ[i] = clamp.(ϕ * μ[i], 0.0, opts.penalty_max)
	end
end


function reset!(con::ConstraintVals{T,W,C,P}, opts::StaticALSolverOptions{T}) where {T,W,C,P}
	λ = con.λ
	c = con.vals
	μ = con.μ
	for i in eachindex(con.inds)
		μ[i] = opts.penalty_initial * @SVector ones(T,P)
		c[i] *= 0.0
		λ[i] *= 0.0
	end
end
