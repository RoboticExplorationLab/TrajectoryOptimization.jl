

function remove_bounds!(conSet::ConstraintSets)
    bnds = filter(is_bound, conSet.constraints)
    n,m = conSet.n, conSet.m
    filter!(x->!is_bound(x), conSet.constraints)
    num_constraints!(conSet)  # re-calculate number of constraints after removing bounds
	return bnds
end

@inline remove_goals!(conSet::ConstraintSets) = remove_constraint_type!(conSet, GoalConstraint)

function remove_constraint_type!(conSet::ConstraintSets, ::Type{Con}) where Con <: AbstractStaticConstraint
	goals = filter(x->x.con isa Con, conSet.constraints)
	filter!(x->!(x.con isa Con), conSet.constraints)
    num_constraints!(conSet)  # re-calculate number of constraints after removing goals
	return goals
end


function get_bounds(conSet::ConstraintSets)
    N = length(conSet.p)

	bnds = remove_bounds!(conSet)
	n,m = size(conSet)
    inds = [bnd.inds for bnd in bnds]

    # Make sure the bounds don't overlap
    if !isempty(bnds)
        Ltotal = mapreduce(length,+,inds)  # total number of knot points covered by bounds
        Linter = length(reduce(union,inds))  # number of knot points with a bound
        @assert Ltotal == Linter
    end

    # Get primal bounds
    zL = [-ones(n+m)*Inf for k = 1:N]
    zU = [ ones(n+m)*Inf for k = 1:N]

    for k = 1:N
        for bnd in bnds
            if k ∈ bnd.inds
                zL[k] = bnd.con.z_min
                zU[k] = bnd.con.z_max
            end
        end
    end

	# Set bounds on goal constraints
	goals = remove_goals!(conSet)
	for goal in goals
		for (i,k) in enumerate(goal.inds)
			zL[k][goal.con.inds] = goal.con.xf
			zU[k][goal.con.inds] = goal.con.xf
		end
	end
	zU = vcat(zU...)
	zL = vcat(zL...)

    # Get bounds for constraints
    p = conSet.p
    NP = sum(p)
    gL = -Inf*ones(NP)
    gU =  Inf*ones(NP)
    cinds = gen_con_inds(conSet)
    for k = 1:N
        for (i,con) in enumerate(conSet.constraints)
            if k ∈ con.inds
                j = _index(con,k)
                gL[cinds[i][j]] = lower_bound(con)
                gU[cinds[i][j]] = upper_bound(con)
            end
        end
    end

    convertInf!(zU)
    convertInf!(zL)
    convertInf!(gU)
    convertInf!(gL)
    return zU, zL, gU, gL
end

function add_dynamics_constraints!(prob::StaticProblem{Q}) where Q
	n,m = size(prob)
    conSet = prob.constraints

    # Implicit dynamics
    dyn_con = ConstraintVals( DynamicsConstraint{Q}(prob.model, prob.N), 1:prob.N-1 )
    add_constraint!(conSet, dyn_con, 1)

    # Initial condition
    init_con = ConstraintVals( GoalConstraint(n,m,prob.x0), 1:1)
    add_constraint!(conSet, init_con, 1)

    return nothing
end

function cost(solver::StaticDIRCOLSolver{Q}) where Q<:QuadratureRule
	Z = get_trajectory(solver)
	obj = get_objective(solver)
	cost(obj, solver.dyn_con, Z)
end

function cost(obj, dyn_con::DynamicsConstraint{HermiteSimpson}, Z)
	N = length(Z)
	model = dyn_con.model
    xMid = dyn_con.xMid
	fVal = dyn_con.fVal
	for k = 1:N
		fVal[k] = dynamics(model, Z[k])
	end
	for k = 1:N-1
		xMid[k] = (state(Z[k]) + state(Z[k+1]))/2 + Z[k].dt/8 * (fVal[k] - fVal[k+1])
	end
	J = 0.0
	for k = 1:N-1
		Um = (control(Z[k]) + control(Z[k+1]))*0.5
		J += Z[k].dt/6 * (stage_cost(obj[k], state(Z[k]), control(Z[k])) +
					    4*stage_cost(obj[k], xMid[k], Um) +
					      stage_cost(obj[k], state(Z[k+1]), control(Z[k+1])))
	end
	J += stage_cost(obj[N], state(Z[N]))
	return J
end

function cost_gradient!(solver::StaticDIRCOLSolver{Q}) where Q
	obj = get_objective(solver)
	Z = get_trajectory(solver)
	dyn_con = solver.dyn_con
	E = solver.E
	cost_gradient!(E, obj, dyn_con, Z)
end

function cost_gradient!(E, obj, dyn_con::DynamicsConstraint{HermiteSimpson}, Z)
	N = length(Z)
	xi = Z[1]._x
	ui = Z[1]._u

	model = dyn_con.model
	fVal = dyn_con.fVal
	xMid = dyn_con.xMid
	∇f = dyn_con.∇f

	for k = 1:N
		fVal[k] = dynamics(model, Z[k])
	end
	for k = 1:N-1
		xMid[k] = (state(Z[k]) + state(Z[k+1]))/2 + Z[k].dt/8 * (fVal[k] - fVal[k+1])
	end
	for k = 1:N
		∇f[k] = jacobian(model, Z[k])
		E.x[k] *= 0
		E.u[k] *= 0
	end

	for k in 1:N-1
		x1, u1 = state(Z[k]),   control(Z[k])
		x2, u2 = state(Z[k+1]), control(Z[k+1])
		xm, um = xMid[k], 0.5*(u1 + u2)
		Fm = jacobian(model, [xm; um])
		A1 = ∇f[k][xi,xi]
		B1 = ∇f[k][xi,ui]
		Am = Fm[xi,xi]
		Bm = Fm[xi,ui]
		A2 = ∇f[k+1][xi,xi]
		B2 = ∇f[k+1][xi,ui]
		dt = Z[k].dt

		∇x1,∇u1 = gradient(obj[k], x1, u1)
		∇x2,∇u2 = gradient(obj[k], x2, u2)
		∇xm,∇um = gradient(obj[k], xm, um)

		E.x[k]   += dt/6 * (∇x1 + 4*( dt/8 * A1 + I/2)'∇xm)
		E.u[k]   += dt/6 * (∇u1 + 4*( ( dt/8 * B1)'∇xm + 0.5I'*∇um))
		E.x[k+1] += dt/6 * (∇x2 + 4*(-dt/8 * A2 + I/2)'∇xm)
		E.u[k+1] += dt/6 * (∇u2 + 4*( (-dt/8 * B2)'∇xm + 0.5I'*∇um))
	end

	E.x[N] += gradient(obj[N], state(Z[N]), control(Z[N]))[1]
	return nothing
end

function copy_gradient!(grad_f, E::CostExpansion, xinds, uinds)
    for k = 1:length(uinds)
        grad_f[xinds[k]] = E.x[k]
        grad_f[uinds[k]] = E.u[k]
    end
    if length(xinds) != length(uinds)
        grad_f[xinds[end]] = E.x[end]
    end
end
