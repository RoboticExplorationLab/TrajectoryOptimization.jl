
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
				cons[i][_index(con,k)] = idx .+ (1:conLen[i])
				idx += conLen[i]
	        end
	    end
	elseif structure == :by_knotpoint
		for k = 1:N
			for (i,con) in enumerate(conSet.constraints)
				if k in con.inds
					j = _index(con,k)
					cons[i][j] = idx .+ (1:conLen[i])
					idx += conLen[i]
				end
			end
		end
	end
    return cons
end

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
					j = _index(con,k)
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


#############################################################################################
#                             COPY TO BLOCK MATRICES                                       #
#############################################################################################

# Copy Constraints
function copy_inds(dest, src, inds)
    for i in eachindex(inds)
        dest[inds[i]] = src[i]
    end
end

function copy_constraints!(d, solver::DirectSolver)
    conSet = get_constraints(solver)
    for (i,con) in enumerate(conSet.constraints)
        copy_inds(d, con.vals, solver.con_inds[i])
    end
    return nothing
end

function copy_active_set!(a, solver::DirectSolver)
    conSet = get_constraints(solver)
    for i = 1:length(conSet.constraints)
        copy_inds(solver.active_set, conSet.constraints[i].active, solver.con_inds[i])
    end
end


"""Copy constraint Jacobians to given indices in a sparse array
Dispatches on bandedness of the constraint"""
function copy_jacobian!(D, con::ConstraintVals{T,Stage}, cinds, xinds, uinds) where T
    for (i,k) in enumerate(con.inds)
        zind = [xinds[k]; uinds[k]]
        D[cinds[i], zind] .= con.∇c[i]
    end
end

function copy_jacobian!(D, con::ConstraintVals{T,State}, cinds, xinds, uinds) where T
    for (i,k) in enumerate(con.inds)
        D[cinds[i], xinds[k]] .= con.∇c[i]
    end
end

function copy_jacobian!(D, con::ConstraintVals{T,Control}, cinds, xinds, uinds) where T
    for (i,k) in enumerate(con.inds)
        D[cinds[i], uinds[k]] .= con.∇c[i]
    end
end

function copy_jacobian!(D, con::ConstraintVals{T,Coupled}, cinds, xinds, uinds) where T
    for (i,k) in enumerate(con.inds)
        zind = [xinds[k]; uinds[k]; xinds[k+1]; uinds[k+1]]
        D[cinds[i], zind] .= con.∇c[i]
    end
end

function copy_jacobian!(D, con::Union{ConstraintVals{T,Dynamical}, ConstraintVals{T,Coupled,<:DynamicsConstraint{Q}}},
		cinds, xinds, uinds) where {T,Q<:Implicit}
    for (i,k) in enumerate(con.inds)
        zind = [xinds[k]; uinds[k]; xinds[k+1]]
        D[cinds[i], zind] .= con.∇c[i]
    end
end

"Copy all constraint Jacobians to a sparse matrix"
function copy_jacobians!(D, solver::DirectSolver)
    conSet = get_constraints(solver)
    xinds, uinds = primal_partition(solver)
    cinds = solver.con_inds

    for i = 1:length(conSet.constraints)
        copy_jacobian!(D, conSet.constraints[i], cinds[i], xinds, uinds)
    end
    return nothing
end


"Copy constraint Jacobians to linear indices of a vector"
function copy_jacobian!(d::AbstractVector{<:Real}, con::ConstraintVals, linds)
	for (j,k) in enumerate(con.inds)
		inds = linds[j]
		d[inds] = con.∇c[j]
	end
end

"Copy all constraint Jacobians to linear indices of a vector"
function copy_jacobians!(jac::AbstractVector{<:Real}, solver::DirectSolver)
    conSet = get_constraints(solver)
    xinds, uinds = primal_partition(solver)
    cinds = solver.con_inds
    linds = jacobian_linear_inds(solver)

    for (i,con) in enumerate(conSet.constraints)
		copy_jacobian!(jac, con, linds[i])
    end
    return nothing
end
