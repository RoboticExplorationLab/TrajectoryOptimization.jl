
function DoubleIntegrator()
    opts = SolverOptions(
        penalty_scaling=1000.,
        penalty_initial=1.,
    )

    model = RobotZoo.DoubleIntegrator()
    n,m = size(model)

    # Task
    x0 = @SVector [0., 0.]
    xf = @SVector [1., 0]
    tf = 2.0

    # Discretization info
    N = 21
    dt = tf/(N-1)

    # Costs
    Q = 1.0*Diagonal(@SVector ones(n))
    Qf = 1.0*Diagonal(@SVector ones(n))
    R = 1.0e-1*Diagonal(@SVector ones(m))
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Constraints
    u_bnd = 3.0
    x_bnd = [Inf,0.6]
    conSet = ConstraintSet(n,m,N)
    bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)
    goal = GoalConstraint(xf)
    add_constraint!(conSet, bnd, 1:N-1)
    add_constraint!(conSet, goal, N:N)

    doubleintegrator_static = Problem(model, obj, xf, tf, constraints=conSet, x0=x0, N=N)
    rollout!(doubleintegrator_static)
    return doubleintegrator_static, opts
end
