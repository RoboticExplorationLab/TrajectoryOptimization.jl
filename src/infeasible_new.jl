
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
