function quadrotor_benchmarks!(suite::BenchmarkGroup)

    suite["line"] = BenchmarkGroup()
    # suite["maze"] = BenchmarkGroup()

    T = Float64

    # options
    max_con_viol = 1.0e-8
    verbose=false

    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
        iterations=300)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
        opts_uncon=opts_ilqr,
        iterations=40,
        cost_tolerance=1.0e-5,
        cost_tolerance_intermediate=1.0e-4,
        constraint_tolerance=max_con_viol,
        penalty_scaling=10.,
        penalty_initial=1.)

    opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
        feasibility_tolerance=max_con_viol,
        solve_type=:feasible)

    opts_altro = ALTROSolverOptions{T}(verbose=verbose,
        opts_al=opts_al,
        R_inf=1.0e-8,
        resolve_feasible_problem=false,
        opts_pn=opts_pn,
        projected_newton=true,
        projected_newton_tolerance=1.0e-3)

    opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=Ipopt.Optimizer(),
        opts=Dict(:max_iter=>10000),
        feasibility_tolerance=max_con_viol)

    opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=SNOPT7.Optimizer(),
        feasibility_tolerance=max_con_viol,
        opts=Dict(:Iterations_limit=>500000,
            :Major_iterations_limit=>1000))
    run_benchmarks!(suite, Problems.quadrotor, [opts_ilqr, opts_al, opts_altro, opts_ipopt, opts_snopt])



    # Quadrotor in Maze
    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
        iterations=300)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
        opts_uncon=opts_ilqr,
        iterations=40,
        cost_tolerance=1.0e-5,
        cost_tolerance_intermediate=1.0e-4,
        constraint_tolerance=max_con_viol,
        penalty_scaling=10.,
        penalty_initial=1.)

    opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
        feasibility_tolerance=max_con_viol,
        solve_type=:feasible)

    opts_altro = ALTROSolverOptions{T}(verbose=verbose,
        opts_al=opts_al,
        R_inf=1.0e-8,
        resolve_feasible_problem=false,
        opts_pn=opts_pn,
        projected_newton=true,
        projected_newton_tolerance=1.0e-4)

    opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=Ipopt.Optimizer(),
        opts=Dict(:max_iter=>10000),
        feasibility_tolerance=1.0e-3)

    opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
        nlp=SNOPT7.Optimizer(),
        feasibility_tolerance=1.0e-3,
        opts=Dict(:Iterations_limit=>500000,
            :Major_iterations_limit=>1000))
    # run_benchmarks!(suite["maze"], Problems.quadrotor_maze, [opts_altro,])

end
