
"$(SIGNATURES) Remove bounds constraints from constraint set"
function remove_bounds!(conSet::ConstraintSet)
    bnds = filter(is_bound, conSet.constraints)
    n,m = conSet.n, conSet.m
    filter!(x->!is_bound(x), conSet.constraints)
    TrajOptCore.num_constraints!(conSet)  # re-calculate number of constraints after removing bounds
	return bnds
end

@inline remove_goals!(conSet::ConstraintSet) = remove_constraint_type!(conSet, GoalConstraint)

"$(SIGNATURES) Remove a type of constraint from constraint set"
function remove_constraint_type!(conSet::ConstraintSet, ::Type{Con}) where Con <: AbstractConstraint
	goals = filter(x->x.con isa Con, conSet.constraints)
	filter!(x->!(x.con isa Con), conSet.constraints)
    TrajOptCore.num_constraints!(conSet)  # re-calculate number of constraints after removing goals
	return goals
end

"$(SIGNATURES) Remove bounds from constraint set and return them as vectors"
function get_bounds(conSet::ConstraintSet)
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
                j = TrajOptCore._index(con,k)
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

"$(SIGNATURES) Generate the indices into the concatenated constraint vector for each constraint.
Determines the bandedness of the Jacobian"
function gen_con_inds(conSet::ConstraintSet, structure=:by_knotpoint)
	n,m = size(conSet)
    N = length(conSet.p)
    numcon = length(conSet.constraints)
    conLen = length.(conSet.constraints)

    cons = [[@SVector ones(Int,length(con)) for i in eachindex(con.inds)] for con in conSet.constraints]

    # Dynamics and general constraints
    idx = 0
	if structure == :by_constraint
	    for (i,con) in enumerate(conSet.constraints)
			for (j,k) in enumerate(con.inds)
				cons[i][TrajOptCore._index(con,k)] = idx .+ (1:conLen[i])
				idx += conLen[i]
	        end
	    end
	elseif structure == :by_knotpoint
		for k = 1:N
			for (i,con) in enumerate(conSet.constraints)
				if k in con.inds
					j = TrajOptCore._index(con,k)
					cons[i][j] = idx .+ (1:conLen[i])
					idx += conLen[i]
				end
			end
		end
	end
    return cons
end

"$(SIGNATURES)
Get the constraint Jacobian structure as a sparse array, and fill in the linear indices
used for filling a vector of the non-zero elements of the Jacobian"
function constraint_jacobian_structure(solver::DirectSolver,
		structure=:by_knopoint)
    n,m,N = size(solver)
    conSet = get_constraints(solver)
    idx = 0.0
    linds = jacobian_linear_inds(solver)

    NN = num_primals(solver)
    NP = num_duals(solver)
    D = spzeros(Int,NP,NN)

    # Number of elements in each block
    blk_len = map(con->length(con.∇c[1]), conSet.constraints)

    # Number of knot points for each constraint
    con_len = map(con->length(con.∇c), conSet.constraints)

    # Linear indices
	if structure == :by_constraint
	    for (i,con) in enumerate(conSet.constraints)
	        for (j,k) in enumerate(con.inds)
	            inds = idx .+ (1:blk_len[i])
	            linds[i][j] = inds
	            con.∇c[j] = inds
	            idx += blk_len[i]
	        end
	    end
	elseif structure == :by_knotpoint
		for k = 1:N
			for (i,con) in enumerate(conSet.constraints)
				if k in con.inds
					inds = idx .+ (1:blk_len[i])
					j = TrajOptCore._index(con,k)
					linds[i][j] = inds
					con.∇c[j] = inds
					idx += blk_len[i]
				end
			end
		end
	end

    copy_jacobians!(D, solver)
    return D
end
