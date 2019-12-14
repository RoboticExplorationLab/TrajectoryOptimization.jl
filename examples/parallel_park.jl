
# options
T = Float64
max_con_viol = 1.0e-8
verbose=false


opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=30,
    penalty_scaling=10.0,
    constraint_tolerance=max_con_viol)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    active_set_tolerance=1e-4,
    feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-4)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    feasibility_tolerance=max_con_viol)

# Solve with AL-iLQR
prob = copy(Problems.parallel_park_static)
U0 = deepcopy(controls(prob))
al = AugmentedLagrangianSolver(prob, opts_al)
initial_controls!(al, U0)
solve!(al)
max_violation(al)

@btime begin
    initial_controls!($al, $U0)
    solve!($al)
end

# Solve with ALTRO
prob = copy(Problems.parallel_park_static)
U0 = deepcopy(controls(prob))
altro = ALTROSolver(prob, opts_altro)
initial_controls!(altro, U0)
solve!(altro)
conSet = get_constraints(altro)

@btime begin
    initial_controls!($altro, $U0)
    solve!($altro)
end
max_violation(altro)

# Ipopt
prob_ipopt = copy(Problems.parallel_park_static)
prob_ipopt = TO.change_integration(prob_ipopt, HermiteSimpson)
rollout!(prob_ipopt)
ipopt = DIRCOLSolver(prob_ipopt, opts_ipopt)
ipopt.opts.verbose = false
solve!(ipopt)
max_violation(ipopt)

@btime solve!($ipopt)
