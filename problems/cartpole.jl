function Cartpole(method=:none)

    opts = SolverOptions(
        cost_tolerance_intermediate=1e-2,
        penalty_scaling=10.,
        penalty_initial=1.0
    )

    model = Dynamics.Cartpole()
    n,m = size(model)
    N = 101
    tf = 5.
    dt = tf/(N-1)

    Q = 1.0e-2*Diagonal(@SVector ones(n))
    Qf = 100.0*Diagonal(@SVector ones(n))
    R = 1.0e-1*Diagonal(@SVector ones(m))
    x0 = @SVector zeros(n)
    xf = @SVector [0, pi, 0, 0]
    obj = LQRObjective(Q,R,Qf,xf,N)

    u_bnd = 3.0
    bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
    goal = GoalConstraint(xf)
    con_bnd = ConstraintVals(bnd, 1:N-1)
    con_goal = ConstraintVals(goal, N:N)
    conSet = ConstraintSet(n,m,[con_bnd, con_goal], N)

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = Traj(X0,U0,dt*ones(N))
    prob = Problem{RK3}(model, obj, conSet, x0, xf, Z, N, 0.0, tf)
    rollout!(prob)

    return prob, opts
end
