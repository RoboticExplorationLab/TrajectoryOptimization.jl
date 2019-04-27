"Create a minimum time problem"
function minimum_time_problem(prob::Problem{T},R_min_time::T=1.0,dt_max::T=1.0,dt_min::T=1.0e-3) where T
    # modify problem with time step control
    N = prob.N; n = prob.model.n; m = prob.model.m
    @assert all([prob.obj[k] isa QuadraticCost for k = 1:N]) #TODO generic cost

    # modify problem with slack control
    obj_mt = CostFunction[]
    for k = 1:N-1
        cost_mt = copy(prob.obj[k])
        cost_mt.Q = cat(cost_mt.Q,0.0,dims=(1,2))
        cost_mt.q = [cost_mt.q; 0.0]
        cost_mt.R = cat(cost_mt.R,2.0*R_min_time,dims=(1,2))
        cost_mt.r = [cost_mt.r; 0.0]
        cost_mt.H = [cost_mt.H zeros(prob.model.m); zeros(prob.model.n+1)']
        push!(obj_mt,cost_mt)
    end
    cost_mt = copy(prob.obj[N])
    cost_mt.Qf = cat(cost_mt.Qf,0.0,dims=(1,2))
    cost_mt.qf = [cost_mt.qf; 0.0]
    push!(obj_mt,cost_mt)

    model_min_time = add_min_time_controls(prob.model)
    con_min_time_eq, con_min_time_bnd = min_time_constraints(n,m,dt_max,dt_min)

    con_prob = AbstractConstraintSet[]
    constrained = is_constrained(prob)
    for k = 1:N-1
        con_mt = AbstractConstraint[]
        constrained ? append!(con_mt,update_constraint_set_jacobians(prob.constraints[k],prob.model.n,prob.model.n+1,prob.model.m)) : nothing
        push!(con_mt,con_min_time_bnd)
        k != 1 ? push!(con_mt,con_min_time_eq) : nothing
        push!(con_prob,con_mt)
    end
    constrained ? push!(con_prob,prob.constraints[N]) : push!(con_prob,Constraint[])

    # return con_prob, obj_mt
    update_problem(prob,model=model_min_time,obj=Objective(obj_mt),
        constraints=ProblemConstraints(con_prob),
        U=[[prob.U[k];sqrt(prob.dt)] for k = 1:prob.N-1],
        X=[[prob.X[k];sqrt(prob.dt)] for k = 1:prob.N],
        x0=[prob.x0;0.0])
end

"Add minimum time controls to dynamics "
function add_min_time_controls(model::Model{M,D}) where {M<:ModelType,D<:Discrete}
    n = model.n; m = model.m
    n̄ = n+1; m̄ = m+1; n̄m̄ = n̄+m̄
    idx = merge(create_partition((m,1),(:u,:mintime)),(x=1:n,))
    idx2 = [idx.x...,(idx.u .+ n̄)...,n̄m̄]

    function f!(x₊::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T},dt::T) where T
        h = u[end]
        model.f(view(x₊,idx.x),x[idx.x],u[idx.u],h^2)
        x₊[n̄] = h
    end

    function ∇f!(Z::AbstractMatrix{T},x::AbstractVector{T},u::AbstractVector{T},dt::T) where T
        h = u[end]
        model.∇f(view(Z,idx.x,idx2),x[idx.x],u[idx.u],h^2)
        Z[idx.x,n̄m̄] .*= 2*h
        Z[n̄,n̄m̄] = 1.0
    end
    AnalyticalModel{M,Discrete}(f!,∇f!,n̄,m̄,model.r,model.params,model.info)
end

"Return the total duration of trajectory"
function total_time(prob::Problem{T}) where T
    m̄ = prob.model.m + 1
    tt = 0.0
    try
        tt = sum([prob.U[k][m̄] for k = 1:prob.N-1])
    catch
        tt = prob.dt*(prob.N-1)
    end
    return tt
end
