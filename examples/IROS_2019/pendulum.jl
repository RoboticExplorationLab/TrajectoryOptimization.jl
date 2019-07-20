using BenchmarkTools, Plots, SNOPT7

T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,
    cost_tolerance_intermediate=1.0e-3,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3,
    opts_al=opts_al,
    opts_pn=opts_pn)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    feasibility_tolerance=max_con_viol)

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    feasibility_tolerance=max_con_viol)

# ALTRO w/ Newton
prob_altro = copy(Problems.pendulum)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation_direct(p1)
plot(p1.X,title="Pendulum state (ALTRO)")
plot(p1.U,title="Pendulum control (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.pendulum)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.pendulum) # get continuous time model
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation_direct(p2)
plot(p2.X,title="Pendulum state (Ipopt)")
plot(p2.U,title="Pendulum control (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.pendulum)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.pendulum) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
plot(p3.X,title="Pendulum state (SNOPT)")
plot(p3.U,title="Pendulum control (SNOPT)")

## Minimum Time
dt_max = 0.15
dt_min = 1.0e-3
dt = 0.15/2
R_minimum_time = 15.0

opts_altro_mt = ALTROSolverOptions{T}(verbose=verbose,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3,
    opts_al=opts_al,
    opts_pn=opts_pn,
    R_minimum_time=R_minimum_time,
    dt_max=dt_max,
    dt_min=dt_min)

opts_ipopt_mt = DIRCOLSolverMTOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    opts=Dict(:print_level=>3,
        :tol=>max_con_viol,
        :constr_viol_tol=>max_con_viol),
    R_min_time=R_minimum_time,
    h_max=dt_max,
    h_min=dt_min)

opts_snopt_mt = DIRCOLSolverMTOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    opts=Dict(:Major_print_level=>0,
        :Minor_print_level=>0,
        :Major_optimality_tolerance=>max_con_viol,
        :Major_feasibility_tolerance=>max_con_viol,
        :Minor_feasibility_tolerance=>max_con_viol),
    R_min_time=R_minimum_time,
    h_max=dt_max,
    h_min=dt_min)

# ALTRO w/ Newton
prob_altro_mt = update_problem(copy(Problems.pendulum),dt=dt,tf=0.)
@time p4, s4 = solve(prob_altro_mt, opts_altro_mt)
@benchmark p4, s4 = solve($prob_altro_mt, $opts_altro_mt)
max_violation_direct(p4)
total_time(p4)
plot(p4.X,title="Pendulum state (Min. Time) (ALTRO)")
plot(p4.U,title="Pendulum control (Min. Time) (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt_mt = copy(Problems.pendulum)
rollout!(prob_ipopt_mt)
prob_ipopt_mt = update_problem(prob_ipopt_mt,model=Dynamics.pendulum,dt=dt,tf=0.) # get continuous time model
@time p5, s5 = solve(prob_ipopt_mt, opts_ipopt_mt)
@benchmark p5, s5 = solve($prob_ipopt_mt, $opts_ipopt_mt)
max_violation_direct(p5)
total_time(p5)
U5 = to_array([p5.U[k][1:p5.model.m] for k = 1:p5.N])
plot(p5.X,title="Pendulum state (Min. Time) (Ipopt)")
plot(U5',title="Pendulum control (Min. Time) (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt_mt = copy(Problems.pendulum)
rollout!(prob_snopt_mt)
prob_snopt_mt = update_problem(prob_snopt_mt,model=Dynamics.pendulum,dt=dt,tf=0.) # get continuous time model
@time p6, s6 = solve(prob_snopt_mt, opts_snopt_mt)
@benchmark p6, s6 = solve($prob_snopt_mt, $opts_snopt_mt)
max_violation_direct(p6)
total_time(p6)
U6 = to_array([p6.U[k][1:p6.model.m] for k = 1:p6.N])
plot(p6.X,title="Pendulum state (Min. Time) (SNOPT)")
plot(U6',title="Pendulum control (Min. Time) (SNOPT)")
