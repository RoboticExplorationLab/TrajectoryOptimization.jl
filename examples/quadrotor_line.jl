using Ipopt, BenchmarkTools

# options
T = Float64
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = StaticiLQRSolverOptions{T}(verbose=verbose,
    iterations=300)

opts_al = StaticALSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)

opts_pn = StaticPNSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = StaticALTROSolverOptions{T}(verbose=verbose,
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

# iLQR
prob = copy(Problems.quadrotor_static)
U0 = deepcopy(controls(prob))
ilqr = StaticiLQRSolver(prob, opts_ilqr)
initial_controls!(ilqr, U0)
solve!(ilqr)
ilqr.stats.iterations

@btime begin
    initial_controls!($ilqr, $U0)
    solve!($ilqr)
end
# 1.25 ms/iteration

# AL-iLQR
prob = copy(Problems.quadrotor_static)
U0 = deepcopy(controls(prob))
alilqr = StaticALSolver(prob)
initial_controls!(alilqr, U0)
solve!(alilqr)
alilqr.stats.cost
max_violation(alilqr)
cost(alilqr)

@btime begin
    initial_controls!($alilqr, $U0)
    solve!($alilqr)
end

# ALTRO
prob = copy(Problems.quadrotor_static)
altro = StaticALTROSolver(prob, opts_altro)
initial_controls!(altro, U0)
solve!(altro)
max_violation(altro)

@btime begin
    initial_controls!($altro, $U0)
    solve!($altro)
end

# Ipopt
prob_ipopt = copy(Problems.cartpole_static)
prob_ipopt = TO.change_integration(prob_ipopt, HermiteSimpson)
rollout!(prob_ipopt)
opts_ipopt.verbose = true
ipopt = StaticDIRCOLSolver(prob_ipopt, opts_ipopt)
ipopt.optimizer.options
MOI.optimize!(ipopt.optimizer)
solve!(ipopt)
ipopt.stats
max_violation(ipopt)

@btime solve!($ipopt)
