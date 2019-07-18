using BenchmarkTools, Plots, SNOPT7

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
    feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    projected_newton=true,
    projected_newton_tolerance=1.0e-4)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    feasibility_tolerance=max_con_viol)

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    feasibility_tolerance=max_con_viol)


# ALTRO w/ Newton
prob_altro = copy(Problems.parallel_park_problem)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation_direct(p1)
X1 = to_array(p1.X)
plot(X1[1,:],X1[2,:],title="Parallel Park - (ALTRO)")
plot(p1.U,title="Parallel Park - (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.parallel_park_problem)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.car_model) # get continuous time model
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation_direct(p2)
X2 = to_array(p2.X)
plot(X2[1,:],X2[2,:],title="Parallel Park - (Ipopt)")
plot(p2.U,title="Parallel Park - (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.parallel_park_problem)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.car_model) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
X3 = to_array(p3.X)
plot(X3[1,:],X3[2,:],title="Parallel Park - (SNOPT)")
plot(p3.U,title="Parallel Park - (SNOPT)")

## Minimum Time
max_con_viol = 1.0e-6
dt_max = 0.2
dt_min = 1.0e-3
R_min_time = 12.5
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=30,
    penalty_scaling=10.0,
    constraint_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,R_minimum_time=R_min_time,
    dt_max=dt_max,
    dt_min=dt_min,
    projected_newton=true,
    projected_newton_tolerance=1.0e-5)

opts_mt_ipopt = TO.DIRCOLSolverMTOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    opts=Dict(:print_level=>3,
        :tol=>max_con_viol,
        :constr_viol_tol=>max_con_viol),
    R_min_time=R_min_time,
    h_max=dt_max,
    h_min=dt_min)

opts_mt_snopt = TO.DIRCOLSolverMTOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    opts=Dict(:Major_print_level=>0,
        :Minor_print_level=>0,
        :Major_optimality_tolerance=>max_con_viol,
        :Major_feasibility_tolerance=>max_con_viol,
        :Minor_feasibility_tolerance=>max_con_viol),
    R_min_time=R_min_time,
    h_max=dt_max,
    h_min=dt_min)

# ALTRO w/ Newton
prob_mt_altro = update_problem(copy(Problems.parallel_park_problem),tf=0.) # make minimum time problem by setting tf = 0
initial_controls!(prob_mt_altro,copy(p1.U))
@time p4, s4 = solve(prob_mt_altro,opts_altro)
@benchmark p4, s4 = solve($prob_mt_altro, $opts_altro)
max_violation_direct(p4)
total_time(p4)
X4 = to_array(p4.X)
plot(X4[1,:],X4[2,:],title="Parallel Park Min. Time - (ALTRO)")
plot(p4.U,title="Parallel Park Min. Time - (ALTRO)")

# DIRCOL w/ Ipopt
prob_mt_ipopt = update_problem(copy(Problems.parallel_park_problem))
initial_controls!(prob_mt_ipopt,copy(p2.U))
rollout!(prob_mt_ipopt)
prob_mt_ipopt = update_problem(prob_mt_ipopt,model=Dynamics.car_model,tf=0.) # get continuous time model
@time p5, s5 = solve(prob_mt_ipopt, opts_mt_ipopt)
@benchmark p5, s5 = solve($prob_mt_ipopt, $opts_mt_ipopt)
max_violation_direct(p5)
total_time(p5)
X5 = to_array(p5.X)
U5 = to_array([p5.U[k][1:p5.model.m] for k = 1:p5.N])
U5m = to_array([0.5*(p5.U[k][1:p5.model.m] + p5.U[k+1][1:p5.model.m])  for k = 1:p5.N-1])
plot(X5[1,:],X5[2,:],title="Parallel Park Min. Time - (Ipopt)")
plot(U5',title="Parallel Park Min. Time - (Ipopt)")
plot(U5m',title="Parallel Park Min. Time (control midpoint) - (Ipopt)")

# DIRCOL w/ SNOPT
prob_mt_snopt = copy(Problems.parallel_park_problem)
initial_controls!(prob_mt_snopt,copy(p3.U))
rollout!(prob_mt_snopt)
prob_mt_snopt = update_problem(prob_mt_snopt,model=Dynamics.car_model,tf=0.) # get continuous time model
@time p6, s6 = solve(prob_mt_snopt, opts_mt_snopt)
@benchmark p6, s6 = solve($prob_mt_snopt, $opts_mt_snopt)
max_violation_direct(p6)
total_time(p6)
X6 = to_array(p6.X)
U6 = to_array([p6.U[k][1:p6.model.m] for k = 1:p6.N])
U6m = to_array([0.5*(p6.U[k][1:p6.model.m] + p6.U[k+1][1:p6.model.m]) for k = 1:p6.N-1])
plot(X6[1,:],X6[2,:],title="Parallel Park Min. Time - (SNOPT)")
plot(U6',title="Parallel Park Min. Time - (SNOPT)")
plot(U6m',title="Parallel Park Min. Time - (control midpoint) (SNOPT)")





###############################################
#            Create PGF Plot                  #
###############################################

function trajectory_plot_flip(prob::Problem; kwargs...)
    x = [x[2] for x in prob.X]
    y = [x[1] for x in prob.X]
    PGF.Plots.Linear(x,y; kwargs...)
end

t1 = trajectory_plot_flip(p1, mark="none", legendentry="original", style="very thick, color=$col_altro, mark options={fill=$col_altro}");
t2 = trajectory_plot_flip(p2, mark="none", legendentry="Ipopt", style="very thick, color=$col_ipopt, mark options={fill=$col_ipopt}");
t3 = trajectory_plot_flip(p3, mark="none", legendentry="SNOPT", style="very thick, color=$col_snopt, mark options={fill=$col_snopt}");
t4 = trajectory_plot_flip(p4, mark="none", legendentry="mintime", style="very thick, color=black, mark options={fill=$col_altro}, dashed");
t5 = trajectory_plot_flip(p5, mark="none", legendentry="Ipopt", style="very thick, color=black, mark options={fill=$col_ipopt}, dashed");
t6 = trajectory_plot_flip(p6, mark="none", legendentry="SNOPT", style="very thick, color=$col_snopt, mark options={fill=$col_snopt}, dashed");

x0 = prob_altro.x0
xf = prob_altro.xf
goal = ([x0[2], xf[2]],
        [x0[1], xf[1]])
z = ["start","end"]
g = PGF.Plots.Scatter(goal[1], goal[2], z,
    scatterClasses="{start={yellow, mark=*, yellow, scale=2},
        end={mark=square*, red, scale=2}}",
    legendentry=["start", "end"]);

a = Axis([t3; t2; t1; t6; t5; t4; g],
    xmin=-0.1, ymin=-0.3, xmax=1.1, ymax=0.3,
    axisEqualImage=true,
    legendPos="outer north east",
    hideAxis=true)

a = Axis([t1; t4; g],
    xmin=-0.1, ymin=-0.3, xmax=1.1, ymax=0.3,
    axisEqualImage=true,
    legendPos="outer north east",
    hideAxis=true)

# Save to tikz format
# NOTE: To fix the problem with the legend for the start and goal points, replace \addplot+ with \addplot in the tikz file
save(joinpath(paper,"ppark_traj.tikz"), a, include_preamble=false)
