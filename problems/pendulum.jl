import TrajectoryOptimization.AbstractSolver

function Pendulum()
    opts = SolverOptions(
        penalty_scaling=100.,
        penalty_initial=0.1,
    )

    model = Dynamics.Pendulum()
    n,m = size(model)

    # cost
    Q = 1e-3*Diagonal(@SVector ones(n))
    R = 1e-3*Diagonal(@SVector ones(m))
    Qf = 1e-0*Diagonal(@SVector ones(n))
    x0 = @SVector zeros(n)
    xf = @SVector [pi, 0.0]  # i.e. swing up
    obj = LQRObjective(Q,R,Qf,xf,N)

    # constraints
    u_bnd = 3.
    bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
    goal_con = GoalConstraint(xf)

    con_bnd = ConstraintVals(bnd, 1:N-1)
    con_xf = ConstraintVals(goal_con, N:N)
    conSet = ConstraintSet(n,m,[con_bnd, con_xf],N)

    # problem
    U = [@SVector fill(0.1, m) for k = 1:N-1]
    pendulum_static = Problem(model, obj, xf, tf, constraints=conSet, x0=x0)
    initial_controls!(pendulum_static, U)
    return pendulum_static, opts
end
