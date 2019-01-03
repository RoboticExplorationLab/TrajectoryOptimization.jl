# Set up
import TrajectoryOptimization._solve


function pendulum_benchmarks()
    group = BenchmarkGroup()
    stat_group = BenchmarkGroup()

    stats = Dict("iterations"=>0, "setup_time"=>0.0, "c_max"=>Float64, "major iterations"=>0.0, "cost"=>Float64[])
    stats = BenchmarkGroup(["stats"],stats)

    # Params
    disable_logging(Logging.Warn)
    N = 501
    integration = :rk3
    model = Dynamics.pendulum[1]
    n,m = model.n, model.m

    # Objective
    x0 = [0; 0.]
    xf = [pi; 0] # (ie, swing up)
    Q = 1e-3*Diagonal(I,n)
    Qf = 100.0*Diagonal(I,n)
    R = 1e-2*Diagonal(I,m)
    tf = 5.
    obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

    u_min = [-3]
    u_max = [3]
    x_min = [-5;-5]
    x_max = [10; 10]
    obj_con = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

    # Unconstrained
    solver_uncon = Solver(model, obj_uncon, N=N, integration=integration)
    U0 = ones(m,N)

    stat_group["unconstrained"] = deepcopy(stats)
    group["unconstrained"] = @benchmarkable _solve($solver_uncon, $U0, bmark_stats=$stat_group["unconstrained"])

    # Constrained
    solver_con = Solver(model, obj_con, N=N, integration=integration)
    stat_group["constrained"] = deepcopy(stats)
    group["constrained"] = @benchmarkable _solve($solver_con, $U0, bmark_stats=$stat_group["constrained"])


    return group, stat_group
end
