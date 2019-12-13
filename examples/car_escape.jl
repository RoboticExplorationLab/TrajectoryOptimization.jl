const TO = TrajectoryOptimization

# Car escape
T = Float64

# options
max_con_viol = 1.0e-8
verbose=false


opts_al = StaticALSolverOptions{T}(verbose=verbose,
    cost_tolerance=1.0e-6,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=10.)

opts_pn = StaticPNSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = StaticALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-1,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    feasibility_tolerance=max_con_viol)


prob = copy(Problems.car_escape_static)

# ALTRO
altro = StaticALTROSolver(prob, opts_altro, infeasible=true)
Z0 = copy(get_trajectory(altro))
initial_trajectory!(altro, Z0)
solve!(altro)
max_violation(altro)

@btime begin
    initial_trajectory!($altro, $Z0)
    solve!($altro)
end
max_violation(altro)


# Ipopt
prob_ipopt = copy(Problems.car_escape_static)
prob_ipopt = TO.change_integration(prob_ipopt, HermiteSimpson)
ipopt = StaticDIRCOLSolver(prob_ipopt, opts_ipopt)
ipopt.opts.verbose = true
solve!(ipopt)
max_violation(ipopt)
Problems.plot_escape(states(ipopt))

@btime solve!($ipopt)
