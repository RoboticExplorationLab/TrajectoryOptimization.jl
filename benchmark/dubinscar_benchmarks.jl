# Set up
import TrajectoryOptimization._solve

function dubinscar_benchmarks()
    group = BenchmarkGroup()
    stats = BenchmarkGroup()
    group["parallel park"], stats["parallel park"] = parallelpark_benchmark()
    return group, stats
end

function parallelpark_benchmark()
    group = BenchmarkGroup()
    stat_group = BenchmarkGroup()

    stats = Dict("iterations"=>0, "setup_time"=>0.0, "c_max"=>0.0, "major iterations"=>0.0, "cost"=>0.0)
    stats = BenchmarkGroup(["stats"],stats)

    # Params
    disable_logging(Logging.Warn)
    N = 501
    integration = :rk3
    model, obj = Dynamics.dubinscar
    n,m = model.n, model.m

    # Objective
    x0 = [0.0;0.0;0.]
    xf = [0.0;1.0;0.]
    tf =  3.
    Qf = 100.0*Diagonal(I,n)
    Q = (1e-3)*Diagonal(I,n)
    R = (1e-2)*Diagonal(I,m)
    obj = LQRObjective(Q, R, Qf, tf, x0, xf)

    # Unconstrained
    solver_uncon = Solver(model, obj, N=N, integration=integration)
    U0 = ones(m,N)

    stat_group["unconstrained"] = deepcopy(stats)
    group["unconstrained"] = @benchmarkable _solve($solver_uncon, $U0, bmark_stats=$stat_group["unconstrained"])

    # Constrained
    x_min = [-0.25; -0.001; -Inf]
    x_max = [0.25; 1.001; Inf]
    obj_con = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

    solver_con = Solver(model, obj_con, N=N, integration=integration)
    stat_group["constrained"] = deepcopy(stats)
    group["constrained"] = @benchmarkable _solve($solver_con, $U0, bmark_stats=$stat_group["constrained"])


    return group, stat_group
end
