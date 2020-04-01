function Acrobot()
    opts = SolverOptions(
        cost_tolerance_intermediate=1e-2,
        penalty_scaling = 1000.,
        penalty_initial = 0.001,
    )
    # model
    model = RobotZoo.Acrobot()
    n,m = size(model)

    # discretization
    tf = 5.0
    N = 101

    # initial and final conditions
    x0 = @SVector [-pi/2, 0, 0, 0]
    xf = @SVector [+pi/2, 0, 0, 0]

    # objective
    Q = Diagonal(@SVector fill(1.0, n))
    R = Diagonal(@SVector fill(0.01, m))
    Qf = 100*Q
    obj = LQRObjective(Q,R,Qf,xf,N)

    # constraints
    conSet = ConstraintList(n,m,N)
    goal = GoalConstraint(xf)
    bnd  = BoundConstraint(n,m, u_min=-15, u_max=15)
    add_constraint!(conSet, goal, N:N)
    add_constraint!(conSet, bnd, 1:N-1)

    # initialization
    u0 = @SVector fill(0.0,m)

    # set up problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
    rollout!(prob)

    return prob, opts
end
