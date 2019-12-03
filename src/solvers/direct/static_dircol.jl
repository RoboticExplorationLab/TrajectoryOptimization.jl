
function gen_con_inds(conSet::ConstraintSets)
    n,m = size(conSet.constraints[1])
    N = length(conSet.p)
    numcon = length(conSet.constraints)
    conLen = length.(conSet.constraints)

    dyn = [@SVector ones(Int,n) for k = 1:N]
    cons = [[@SVector ones(Int,length(con)) for i in eachindex(con.inds)] for con in conSet.constraints]

    # Initial condition
    dyn[1] = 1:n
    idx = n

    # Dynamics and general constraints
    if has_dynamics(conSet)
        idx = 0
    end
    for k = 1:N-1
        if !has_dynamics(conSet)
            dyn[k+1] = idx .+ (1:n)
            idx += n
        end
        for (i,con) in enumerate(conSet.constraints)
            if k ∈ con.inds
                cons[i][_index(con,k)] = idx .+ (1:conLen[i])
                idx += conLen[i]
            end
        end
    end

    # Terminal constraints
    for (i,con) in enumerate(conSet.constraints)
        if N ∈ con.inds
            cons[i][_index(con,N)] = idx .+ (1:conLen[i])
            idx += conLen[i]
        end
    end
    if has_dynamics(conSet)
        dyn_con_inds = findfirst(x->x.con isa DynamicsConstraint, conSet.constraints)
        dyn = copy(cons[dyn_con_inds])
        init_con_inds = findfirst(x->x.con isa GoalConstraint, conSet.constraints)
        insert!(dyn, 1, cons[init_con_inds][1])
    end

    # return dyn
    return dyn,cons
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

function add_dynamics_constraints!(prob::StaticProblem)
    conSet = get_constraints(prob)

    # Implicit dynamics
    dyn_con = ConstraintVals( ImplicitDynamics(prob.model, prob.N), 1:prob.N-1 )
    add_constraint!(conSet, dyn_con, 1)

    # Initial condition
    init_con = ConstraintVals( GoalConstraint(prob.x0), 1:1)
    add_constraint!(conSet, init_con, 1)
end
