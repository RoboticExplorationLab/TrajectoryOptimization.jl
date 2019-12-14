using BenchmarkTools
using Ipopt
const TO = TrajectoryOptimization


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

# AL-iLQR
prob = copy(Problems.doubleintegrator_static)
U0 = deepcopy(controls(prob))
alilqr = AugmentedLagrangianSolver(prob, opts_al)
solve!(alilqr)
max_violation(alilqr)

@btime begin
    initial_controls!($alilqr, $U0)
    solve!($alilqr)
end


# ALTRO
altro = ALTROSolver(prob, opts_altro)
solve!(altro)
max_violation(altro)

@btime begin
    initial_controls!($altro, $U0)
    solve!($altro)
end

# Ipopt
prob_ipopt = copy(Problems.doubleintegrator_static)
prob_ipopt = TO.change_integration(prob_ipopt, HermiteSimpson)
rollout!(prob_ipopt)
ipopt = DIRCOLSolver(prob_ipopt)
ipopt.opts.verbose = false
solve!(ipopt)
max_violation(ipopt)

@btime solve!($ipopt)
