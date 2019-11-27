
function collocation_constraints!(g, prob::Problem, solver::StaticDIRCOLSolver{T,HermiteSimpson})
    fVal = solver.fVal
    Xm = solver.Xm
    Z = prob.Z
    dinds = solver.dyn_inds

    N = length(Z)
    for k in eachindex(Z)
        fVal[k] = dynamics(prob.model, Z[k])
    end
    for k = 1:N-1
        dt = Z[k].dt
        Xm[k] = (state(Z[k+1]) + state(Z[k]))/2 + dt/8*(fVal[k] - fVal[k+1])
    end
    for k = 1:N-1
        dt = Z[k].dt
        fValm = dynamics(prob.model, Xm[k], ( control(Z[k]) + control(Z[k+1]) )/2 )
        g[dinds[k]] = -state(Z[k+1]) + state(Z[k+1]) + dt*(fVal[k] + 4*fValm + fVal[k+1])/6
    end
end





function get_bounds(conSet::ConstraintSets)

    N = length(conSet.p)

    # Remove bound constraints
    bnds = filter(is_bound, conSet.constraints)
    filter!(x->!is_bound(x), conSet.constraints)
    num_constraints!(conSet)  # re-calculate number of constraints after removing bounds
    inds = [bnd.inds for bnd in bnds]

    # Make sure the bounds don't overlap
    if !isempty(bnds)
        Ltotal = mapreduce(length,+,inds)  # total number of knot points covered by bounds
        Linter = length(reduce(union,inds))  # number of knot points with a bound
        @assert Ltotal == Linter
    end

    # Get lower bounds
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
    zU = vcat(zU...)
    zL = vcat(zL...)

    # Get bounds for constraints
    p = conSet.p
    NP = sum(p) + N*n
    gL = -Inf*ones(NP)
    gU =  Inf*ones(NP)
    dinds, cinds = gen_con_inds(conSet)
    for k = 1:N
        for (i,con) in enumerate(conSet.constraints)
            if k ∈ con.inds
                j = _index(con,k)
                gL[cinds[i][j]] = lower_bound(con)
                gU[cinds[i][j]] = upper_bound(con)
            end
        end
        gL[dinds[k]] .= 0.0
        gU[dinds[k]] .= 0.0
    end
    return zU, zL, gU, gL
end
