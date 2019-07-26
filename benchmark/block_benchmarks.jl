
function block_benchmarks!(suite::BenchmarkGroup)

    T = Float64

    # options
    max_con_viol = 1.0e-8
    verbose=false

    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=1000.,
    penalty_initial=1.)

    opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    projected_newton=false)

    opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    feasibility_tolerance=max_con_viol)

    opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=SNOPT7.Optimizer(),
    feasibility_tolerance=max_con_viol)

    run_benchmarks!(suite, Problems.doubleintegrator, [opts_ilqr, opts_al, opts_altro, opts_ipopt, opts_snopt])
end
