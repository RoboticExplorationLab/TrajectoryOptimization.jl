
function ParallelPark(method=:none)

    opts = SolverOptions(
        cost_tolerance_intermediate=1e-3,
        active_set_tolerance=1e-4
    )

    # model
    model = RobotZoo.DubinsCar()
    n,m = size(model)
    N = 101
    tf = 3.

    # cost
    x0 = @SVector [0., 0., 0.]
    xf = @SVector [0., 1., 0.]
    Qf = 100.0*Diagonal(@SVector ones(n))
    Q = (1e-2)*Diagonal(@SVector ones(n))
    R = (1e-2)*Diagonal(@SVector ones(m))

    # constraints
    u_bnd = 2.
    x_min = @SVector [-0.25, -0.001, -Inf]
    x_max = @SVector [0.25, 1.001, Inf]
    bnd = BoundConstraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd)
    goal = GoalConstraint(xf)

    # Constraint vals
    con_bnd = ConstraintVals(bnd, 1:N-1)
    con_goal = ConstraintVals(goal, N:N)

    # problem
    U = [@SVector fill(0.1,m) for k = 1:N-1]
    obj = LQRObjective(Q,R,Qf,xf,N)

    conSet = ConstraintSet(n,m,[con_bnd, con_goal], N)

    prob = Problem(model, obj, xf, tf, constraints=conSet, x0=x0, U0=U)

    return prob, opts
end
