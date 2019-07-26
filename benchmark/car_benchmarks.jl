function car_benchmarks!(suite::BenchmarkGroup)

    # Car escape
    T = Float64

    # options
    max_con_viol = 1.0e-8
    verbose=false

    suite["parallel_park"] = BenchmarkGroup()
    suite["3obs"] = BenchmarkGroup()
    suite["escape"] = BenchmarkGroup()

    # Parallel Park
    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
        live_plotting=:off)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
        opts_uncon=opts_ilqr,
        iterations=30,
        penalty_scaling=10.0,
        constraint_tolerance=max_con_viol)

    opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
        feasibility_tolerance=max_con_viol)

    opts_altro = ALTROSolverOptions{T}(verbose=verbose,
        opts_al=opts_al,
        projected_newton=true,
        projected_newton_tolerance=1.0e-4)

    opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=Ipopt.Optimizer(),
        feasibility_tolerance=max_con_viol)

    opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=SNOPT7.Optimizer(),
        feasibility_tolerance=max_con_viol)

    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
        live_plotting=:off)
    run_benchmarks!(suite["parallel_park"], Problems.parallel_park, [opts_ilqr, opts_al, opts_altro, opts_ipopt, opts_snopt])


    # Three Obstacle
    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
        opts_uncon=opts_ilqr,
        cost_tolerance=1.0e-4,
        cost_tolerance_intermediate=1.0e-2,
        constraint_tolerance=max_con_viol,
        penalty_scaling=50.,
        penalty_initial=10.)

    opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
        feasibility_tolerance=max_con_viol)

    opts_altro = ALTROSolverOptions{T}(verbose=verbose,
        opts_al=opts_al,
        opts_pn=opts_pn,
        projected_newton=true,
        projected_newton_tolerance=1.0e-3)

    opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=Ipopt.Optimizer(),
        feasibility_tolerance=max_con_viol)

    opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=SNOPT7.Optimizer(),
        feasibility_tolerance=max_con_viol)

    run_benchmarks!(suite["3obs"], Problems.car_3obs, [opts_altro, opts_al, opts_ipopt, opts_snopt])


    # Car escape
    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
        opts_uncon=opts_ilqr,
        cost_tolerance=1.0e-6,
        cost_tolerance_intermediate=1.0e-2,
        constraint_tolerance=max_con_viol,
        penalty_scaling=50.,
        penalty_initial=10.)

    opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
        feasibility_tolerance=max_con_viol,
        solve_type=:feasible)

    opts_altro = ALTROSolverOptions{T}(verbose=verbose,
        opts_al=opts_al,
        R_inf=1.0e-1,
        resolve_feasible_problem=false,
        opts_pn=opts_pn,
        projected_newton=true,
        projected_newton_tolerance=1.0e-3);

    opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=Ipopt.Optimizer(),
        feasibility_tolerance=max_con_viol)

    opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=SNOPT7.Optimizer(),
        feasibility_tolerance=max_con_viol)

    # run_benchmarks!(suite["escape"], Problems.car_escape, [opts_altro, opts_ipopt, opts_snopt])

end
