
function YakProblems(Rot=UnitQuaternion{Float64}; scenario=:barrellroll, use_rot=Rot<:UnitQuaternion,
        costfun=:Quadratic, normcon=false)
    model = RobotZoo.YakPlane(Rot)
    n,m = size(model)
    rsize = size(model)[1] - 9

    opts = SolverOptions(
        cost_tolerance_intermediate = 1e-1,
        penalty_scaling = 1000.,
        penalty_initial = 0.01
    )

    # Discretization
    tf = 1.25
    N = 101
    dt = tf/(N-1)
    # N = Int(1.25/dt+1)
    # dt = 0.025

    if scenario == :barrellroll
        ey = @SVector [0,1,0.]

        # Initial and final condition
        p0 = MRP(0.997156, 0., 0.075366) # initial orientation
        pf = MRP(0., -0.0366076, 0.) # final orientation (upside down)

        costfun == :QuatLQR ? sq = 0 : sq = 1
        if Rot <: UnitQuaternion
            rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
        else
            rm_quat = @SVector [1,2,3,4,5,6,7,8,9,10,11,12]
        end

        x0 = Dynamics.build_state(model, [-3,0,1.5], p0, [5,0,0], [0,0,0])
        utrim  = @SVector  [41.6666, 106, 74.6519, 106]
        xf = Dynamics.build_state(model, [3,0,1.5], pf, [5,0,0], [0,0,0])

        # Objective
        Qf_diag = Dynamics.fill_state(model, 100, 500*sq, 100, 100.)
        Q_diag = Dynamics.build_state(model, [0.1,0.1,0.1],
            (@SVector fill(0.1*sq,rsize)), fill(0.1,3), fill(0.1,3))
        Qf = Diagonal(Qf_diag)
        Q = Diagonal(Q_diag)
        R = Diagonal(@SVector fill(1e-3,4))
        if costfun == :Quadratic
            costfun = LQRCost(Q, R, xf, utrim)
            costterm = LQRCost(Qf, R, xf, utrim)
        elseif costfun == :QuatLQR
            costfun = QuatLQRCost(Q, R, xf, utrim; w=0.1)
            costterm = QuatLQRCost(Qf, R, xf, utrim; w=200.0)
        elseif costfun == :ErrorQuad
            Q = Diagonal(Q_diag[rm_quat])
            Qf = Diagonal(Qf_diag[rm_quat])
            costfun = ErrorQuadratic(model, Q, R, xf, utrim)
            costterm = ErrorQuadratic(model, Qf, R, xf, utrim)
        end
        obj = Objective(costfun, costterm, N)

        # Constraints
        conSet = ConstraintList(n,m,N)
        goal = GoalConstraint(xf)
        add_constraint!(conSet,goal,N:N)

    else
        throw(ArgumentError("$scenario isn't a known scenario"))
    end

    # Initialization
    U0 = [copy(utrim) for k = 1:N-1]

    # Build problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
    initial_controls!(prob, U0)
    prob, opts
end
