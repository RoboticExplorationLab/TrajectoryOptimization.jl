"Create infeasible state trajectory initialization problem from problem"
function infeasible_problem(prob::Problem{T},R_inf::T=1.0) where T
    N = prob.N
    @assert all([prob.obj[k] isa QuadraticCost for k = 1:N]) #TODO generic cost

    # modify problem with slack control
    obj_inf = CostFunction[]
    for k = 1:N-1
        cost_inf = copy(prob.obj[k])
        cost_inf.R = cat(cost_inf.R,R_inf*Diagonal(I,prob.model.n)/prob.dt,dims=(1,2))
        cost_inf.r = [cost_inf.r; zeros(prob.model.n)]
        cost_inf.H = [cost_inf.H; zeros(prob.model.n,prob.model.n)]
        push!(obj_inf,cost_inf)
    end
    push!(obj_inf,copy(prob.obj[N]))

    model_inf = add_slack_controls(prob.model)
    u_slack = slack_controls(prob)
    con_inf = infeasible_constraints(prob.model.n,prob.model.m)

    con_prob = ConstraintSet[]
    constrained = is_constrained(prob)
    for k = 1:N-1
        _con = AbstractConstraint[]
        constrained ? append!(_con,prob.constraints.C[k]) : nothing
        push!(_con,con_inf)
        push!(con_prob,_con)
    end

    constrained ? push!(con_prob,prob.constraints.C[N]) : push!(con_prob,Constraint[])

    update_problem(prob,model=model_inf,obj=Objective(obj_inf),
        constraints=ProblemConstraints(con_prob),U=[[prob.U[k];u_slack[k]] for k = 1:prob.N-1])
end

"Return a feasible problem from an infeasible problem"
function infeasible_to_feasible_problem(prob::Problem{T},prob_altro::Problem{T},
        state::NamedTuple,opts::ALTROSolverOptions{T}) where T
    prob_altro_feasible = prob

    if state.minimum_time
        prob_altro_feasible = minimum_time_problem(prob_altro_feasible,opts.R_minimum_time,
            opts.dt_max,opts.dt_min)

        # initialize sqrt(dt) from previous solve
        for k = 1:prob.N-1
            prob_altro_feasible.U[k][end] = prob_altro.U[k][end]
            k != 1 ? prob_altro_feasible.X[k][end] = prob_altro.X[k][end] : prob_altro_feasible.X[k][end] = 0.0
        end
        prob_altro_feasible.X[end][end] = prob_altro.X[end][end]
    end

    if opts.dynamically_feasible_projection
        projection!(prob_altro_feasible,opts.opts_al.opts_uncon)
    end

    return prob_altro_feasible
end

"Calculate slack controls that produce infeasible state trajectory"
function slack_controls(prob::Problem{T}) where T
    N = prob.N
    n = prob.model.n
    m = prob.model.m

    dt = prob.dt

    u_slack = [zeros(n) for k = 1:N-1]#zeros(n,N-1)
    x = [zeros(n) for k = 1:N]#zeros(n,N)
    x[1] = prob.x0

    for k = 1:N-1
        evaluate!(x[k+1],prob.model,x[k],prob.U[k],dt)

        u_slack[k] = prob.X[k+1] - x[k+1]
        x[k+1] += u_slack[k]
    end
    return u_slack
end

function line_trajectory(x0::Vector,xf::Vector,N::Int)
    t = range(0,stop=N,length=N)
    slope = (xf .- x0)./N
    x_traj = [slope*t[k] for k = 1:N]
    x_traj[1] = x0
    x_traj[end] = xf
    x_traj
end
