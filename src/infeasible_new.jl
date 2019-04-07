
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


function line_trajectory_new(x0::Vector{T},xf::Vector{T},N::Int) where T
    t = range(0,stop=N,length=N)
    slope = (xf .- x0)./N
    x_traj = [slope*t[k] for k = 1:N]
    x_traj[1] = x0
    x_traj[end] = xf
    x_traj
end
