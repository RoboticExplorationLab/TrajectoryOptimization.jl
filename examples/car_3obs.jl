using Ipopt, StaticArrays, LinearAlgebra, BenchmarkTools, Plots
const TO = TrajectoryOptimization

# Car escape
T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = StaticiLQRSolverOptions{T}(verbose=verbose,
    live_plotting=:off)

opts_al = StaticALSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=10.)

opts_pn = StaticPNSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol)

opts_altro = StaticALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    feasibility_tolerance=max_con_viol)


# AL-iLQR
prob = copy(Problems.car_3obs_static)
U0 = deepcopy(controls(prob))
al = StaticALSolver(prob, opts_al)

initial_controls!(al, U0)
solve!(al)
al.stats.iterations
max_violation(al)

@btime begin
    initial_controls!($al, $U0)
    solve!($al)
end

p = plot(prob.model, get_trajectory(al))
plot!(prob.constraints[1].con)
display(p)



# ALTRO
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
prob_ipopt = copy(Problems.car_3obs_static)
prob_ipopt = TO.change_integration(prob_ipopt, HermiteSimpson)
rollout!(prob_ipopt)
ipopt = StaticDIRCOLSolver(prob_ipopt, opts_ipopt)
ipopt.opts.verbose = true
solve!(ipopt)
max_violation(ipopt)

@btime solve!($ipopt)

p = plot(prob.model, get_trajectory(ipopt))
plot!(prob.constraints[1].con)
display(p)
plot(controls(ipopt))
