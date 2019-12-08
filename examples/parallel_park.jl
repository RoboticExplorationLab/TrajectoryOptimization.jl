
# options
T = Float64
max_con_viol = 1.0e-8
verbose=false


opts_ilqr = StaticiLQRSolverOptions{T}(verbose=verbose,
    live_plotting=:off)

opts_al = StaticALSolverOptions{T}(verbose=verbose,
    opts_uncon=sopts_ilqr,
    iterations=30,
    penalty_scaling=10.0,
    constraint_tolerance=max_con_viol)

opts_pn = StaticPNSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol)

opts_altro = StaticALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-4)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    feasibility_tolerance=max_con_viol)


prob = copy(Problems.parallel_park_static)
U0 = deepcopy(controls(prob))
altro = StaticALTROSolver(sprob, opts_altro)
initial_controls!(altro, U0)
solve!(altro)

@btime begin
    initial_controls!($altro, $U0)
    solve!($altro)
end
max_violation(altro)


# Ipopt
prob_ipopt = copy(Problems.parallel_park_static)
prob_ipopt = TO.change_integration(prob_ipopt, HermiteSimpson)
rollout!(prob_ipopt)
ipopt = StaticDIRCOLSolver(prob_ipopt, opts_ipopt)
ipopt.opts.verbose = false
solve!(ipopt)

@btime solve!($ipopt)
