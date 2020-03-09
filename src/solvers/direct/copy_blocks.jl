

#############################################################################################
#                             COPY TO BLOCK MATRICES                                       #
#############################################################################################

"Copy vectors from a vector of vectors to a concatenated vector"
function copy_inds(dest, src, inds)
    for i in eachindex(inds)
        dest[inds[i]] = src[i]
    end
end

"Copy constraints to a single concatenated vector"
function copy_constraints!(d, solver::DirectSolver)
    conSet = get_constraints(solver)
    for (i,con) in enumerate(conSet.constraints)
        copy_inds(d, con.vals, solver.con_inds[i])
    end
    return nothing
end

"Copy active set to a single concatenated vector"
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

function copy_gradient!(grad_f, E::Vector{<:CostExpansion}, xinds, uinds)
    for k = 1:length(uinds)
        grad_f[xinds[k]] = E[k].x
        grad_f[uinds[k]] = E[k].u
    end
    if length(xinds) != length(uinds)
        grad_f[xinds[end]] = E[end].x
    end
end
