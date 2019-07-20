using BenchmarkTools, Plots, SNOPT7

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
    nlp=:Ipopt,
    feasibility_tolerance=max_con_viol)

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    feasibility_tolerance=max_con_viol)


# ALTRO w/o Newton
prob_altro = copy(Problems.doubleintegrator)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p1)
plot(p1.X,title="Double Integrator state (ALTRO)")
plot(p1.U,title="Double Integrator control (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.doubleintegrator)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.doubleintegrator) # get continuous time model
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation_direct(p2)
plot(p2.X,title="Double Integrator state (Ipopt)")
plot(p2.U,title="Double Integrator control (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.doubleintegrator)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.doubleintegrator) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
plot(p3.X,title="Double Integrator state (SNOPT)")
plot(p3.U,title="Double Integrator control (SNOPT)")
