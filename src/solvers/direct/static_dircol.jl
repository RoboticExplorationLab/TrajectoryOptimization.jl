

function remove_bounds!(conSet::ConstraintSets)
    bnds = filter(is_bound, conSet.constraints)
    n,m = conSet.n, conSet.m
    filter!(x->!is_bound(x), conSet.constraints)
    num_constraints!(conSet)  # re-calculate number of constraints after removing bounds
	return bnds
end

@inline remove_goals!(conSet::ConstraintSets) = remove_constraint_type!(conSet, GoalConstraint)

function remove_constraint_type!(conSet::ConstraintSets, ::Type{Con}) where Con <: AbstractConstraint
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

function add_dynamics_constraints!(prob::Problem{Q}) where Q
	n,m = size(prob)
    conSet = prob.constraints

    # Implicit dynamics
    dyn_con = ConstraintVals( DynamicsConstraint{Q}(prob.model, prob.N), 1:prob.N-1 )
    add_constraint!(conSet, dyn_con, 1)

    # Initial condition
    init_con = ConstraintVals( GoalConstraint(prob.x0), 1:1)
    add_constraint!(conSet, init_con, 1)

    return nothing
end

function cost(solver::DIRCOLSolver{Q}) where Q<:QuadratureRule
	Z = get_trajectory(solver)
	obj = get_objective(solver)
	cost(obj, solver.dyn_con, Z)
end



function cost_gradient!(solver::DIRCOLSolver{Q}) where Q
	obj = get_objective(solver)
	Z = get_trajectory(solver)
	dyn_con = solver.dyn_con
	E = solver.E
	cost_gradient!(E, obj, dyn_con, Z)
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
