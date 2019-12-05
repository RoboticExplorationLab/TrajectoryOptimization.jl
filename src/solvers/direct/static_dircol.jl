
function gen_con_inds(conSet::ConstraintSets)
    n,m = size(conSet.constraints[1])
    N = length(conSet.p)
    numcon = length(conSet.constraints)
    conLen = length.(conSet.constraints)

    cons = [[@SVector ones(Int,length(con)) for i in eachindex(con.inds)] for con in conSet.constraints]

    # Dynamics and general constraints
    idx = 0
    for (i,con) in enumerate(conSet.constraints)
		for (j,k) in enumerate(con.inds)
			cons[i][_index(con,k)] = idx .+ (1:conLen[i])
			idx += conLen[i]
        end
    end
	#
    # # Terminal constraints
    # for (i,con) in enumerate(conSet.constraints)
    #     if N ∈ con.inds
    #         cons[i][_index(con,N)] = idx .+ (1:conLen[i])
    #         idx += conLen[i]
    #     end
    # end

    # return dyn
    return cons
end

function get_bounds(conSet::ConstraintSets)

    N = length(conSet.p)

    # Remove bound constraints
    bnds = filter(is_bound, conSet.constraints)
    n,m = size(bnds[1])
    filter!(x->!is_bound(x), conSet.constraints)
    num_constraints!(conSet)  # re-calculate number of constraints after removing bounds
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

	# Remove Goal Constraints
	goals = filter(x->x.con isa GoalConstraint, conSet.constraints)
	filter!(x->!(x.con isa GoalConstraint), conSet.constraints)
    num_constraints!(conSet)  # re-calculate number of constraints after removing goals
	for goal in goals
		for (i,k) in enumerate(goal.inds)
			zL[k][1:n] = goal.con.xf
			zU[k][1:n] = goal.con.xf
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

function add_dynamics_constraints!(prob::StaticProblem{<:Implicit})
    conSet = get_constraints(prob)

    # Implicit dynamics
    dyn_con = ConstraintVals( ImplicitDynamics(prob.model, prob.N), 1:prob.N-1 )
    add_constraint!(conSet, dyn_con, 1)

    # Initial condition
    init_con = ConstraintVals( GoalConstraint(prob.x0), 1:1)
    add_constraint!(conSet, init_con, 1)

    return nothing
end

function add_dynamics_constraints!(prob::StaticProblem{Q}) where Q<:QuadratureRule
    conSet = get_constraints(prob)

    # Implicit dynamics
    dyn_con = ConstraintVals( ExplicitDynamics{Q}(prob.model, prob.N), 1:prob.N-1 )
    add_constraint!(conSet, dyn_con, 1)

    # Initial condition
    init_con = ConstraintVals( GoalConstraint(prob.x0), 1:1)
    add_constraint!(conSet, init_con, 1)

    return nothing
end

function cost(prob::StaticProblem, solver::DirectSolver)
	# TODO: dispatch on the quadrature rule
	N = prob.N
	Z = prob.Z
    xMid = solver.Xm
	fVal = solver.fVal
	obj = prob.obj
	for k = 1:N
		fVal[k] = dynamics(prob.model, Z[k])
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

function cost_gradient!(prob::StaticProblem, solver::DirectSolver)
	n,m,N = size(prob)
	model = prob.model
	obj = prob.obj

	E = solver.E
	fVal = solver.fVal
	∇f = solver.∇F
	xMid = solver.Xm
	Z = prob.Z

	xi = Z[1]._x
	ui = Z[1]._u

	for k = 1:N
		fVal[k] = dynamics(prob.model, Z[k])
	end
	for k = 1:N-1
		xMid[k] = (state(Z[k]) + state(Z[k+1]))/2 + Z[k].dt/8 * (fVal[k] - fVal[k+1])
	end
	for k = 1:N
		∇f[k] = jacobian(model, Z[k].z)
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

function copy_constraints!(prob::StaticProblem, solver::DirectSolver, d=solver.d)
    conSet = get_constraints(prob)
    for i = 1:length(conSet.constraints)
        copy_inds(d, conSet.constraints[i].vals, solver.con_inds[i])
    end
    return nothing
end


# Copy Constraint Jacobian
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

function copy_jacobian!(D, con::ConstraintVals{T,Dynamical}, cinds, xinds, uinds) where T
    for (i,k) in enumerate(con.inds)
        zind = [xinds[k]; uinds[k]; xinds[k+1]]
        D[cinds[i], zind] .= con.∇c[i]
    end
end

function copy_jacobians!(prob::StaticProblem, solver::DirectSolver, D=solver.D)
    conSet = get_constraints(prob)
    xinds, uinds = primal_partition(solver)
    cinds = solver.con_inds

    for i = 1:length(conSet.constraints)
        copy_jacobian!(D, conSet.constraints[i], cinds[i], xinds, uinds)
    end
    return nothing
end

function copy_jacobian!(d::AbstractVector{<:Real}, con::ConstraintVals, linds)
	for (j,k) in enumerate(con.inds)
		inds = linds[j]
		d[inds] = con.∇c[j]
	end
end

function copy_jacobians!(prob::StaticProblem, solver::DirectSolver, jac::AbstractVector{<:Real})
    conSet = get_constraints(prob)
    xinds, uinds = primal_partition(solver)
    cinds = solver.con_inds
    linds = jacobian_linear_inds(solver)

    for (i,con) in enumerate(conSet.constraints)
		copy_jacobian!(jac, con, linds[i])
    end
    return nothing
end
