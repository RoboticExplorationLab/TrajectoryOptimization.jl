using BenchmarkTools, Plots, SNOPT7

T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=100.,
    penalty_initial=1.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-4);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    feasibility_tolerance=max_con_viol)

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    feasibility_tolerance=max_con_viol)


# ALTRO w/ Newton
prob_altro = copy(Problems.acrobot_problem)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
plot(p1.X,title="Acrobot state (ALTRO)")
plot(p1.U,title="Acrobot control (ALTRO)")
max_violation_direct(p1)

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.acrobot_problem)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.acrobot_model) # get continuous time model
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
plot(p2.X,title="Acrobot state (Ipopt)")
plot(p2.U,title="Acrobot control (Ipopt)")
max_violation_direct(p2)

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.acrobot_problem)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.acrobot_model) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
plot(p3.X,title="Acrobot state (SNOPT)")
plot(p3.U,title="Acrobot control (SNOPT)")
max_violation_direct(p3)
