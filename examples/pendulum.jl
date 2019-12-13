
T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = StaticiLQRSolverOptions{T}(verbose=verbose,
    live_plotting=:off)

opts_al = StaticALSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,
    cost_tolerance_intermediate=1.0e-3,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)

opts_pn = StaticPNSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol)

opts_altro = StaticALTROSolverOptions{T}(verbose=verbose,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3,
    opts_al=opts_al,
    opts_pn=opts_pn)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    feasibility_tolerance=max_con_viol)


prob = copy(Problems.pendulum_static)
U0 = deepcopy(controls(prob))
ilqr = StaticiLQRSolver(prob)
initial_controls!(ilqr, U0)
Z = ilqr.Z
@btime TO.cost!($ilqr.obj, $Z)
ilqr.obj[prob.N-1].Q
solve!(ilqr)

@btime begin
    initial_controls!($ilqr, $U0)
    solve!($ilqr)
end

# Solve with AL-iLQR
prob = copy(Problems.pendulum_static)
U0 = deepcopy(controls(prob))
al = StaticALSolver(prob, opts_al)
initial_controls!(al, U0)
solve!(al)
max_violation(al)

@btime begin
    initial_controls!($al, $U0)
    solve!($al)
end

# Solve with ALTRO
prob = copy(Problems.pendulum_static)
U0 = deepcopy(controls(prob))
altro = StaticALTROSolver(prob, opts_altro)
initial_controls!(altro, U0)
solve!(altro)
max_violation(altro)

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
max_violation(ipopt)

@btime solve!($ipopt)
