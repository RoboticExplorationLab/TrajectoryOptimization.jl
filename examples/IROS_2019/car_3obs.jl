using BenchmarkTools, Plots, SNOPT7
# Car escape
T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=10.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    feasibility_tolerance=max_con_viol)

# opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
#     nlp=SNOPT7.Optimizer(),
#     feasibility_tolerance=max_con_viol)


x0 = Problems.car_3obs.x0
xf = Problems.car_3obs.xf

# AL-iLQR
prob_al= copy(Problems.car_3obs)
@time p0, s0 = solve(prob_al, opts_al)
@btime solve($prob_al, $opts_al)
max_violation_direct(p0)

# ALTRO w/ Newton
prob_altro = copy(Problems.car_3obs)
@time p1, s1 = solve(prob_altro, opts_altro)
@btime solve($prob_altro, $opts_altro)
max_violation_direct(p1)
Problems.plot_car_3obj(p1.X,x0,xf)

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.car_3obs)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.car) # get continuous time model)
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@btime solve($prob_ipopt, $opts_ipopt)
max_violation_direct(p2)
Problems.plot_car_3obj(p2.X,x0,xf)

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.car_3obs)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.car) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
Problems.plot_car_3obj(p3.X,x0,xf)
plot(p3.U)


###############################################
#            Create PGF Plot                  #
###############################################


color_obs= "gray"
style = "color=$color_obs, fill=$color_obs"
p = [PGF.Plots.Circle(circle..., style=style) for circle in Problems.circles_3obs]
t1 = trajectory_plot(p1, mark="*", legendentry="ALTRO", style="very thick, color=$col_altro, mark options={fill=$col_altro}");
t2 = trajectory_plot(p2, mark="*", legendentry="Ipopt", style="very thick, color=$col_ipopt, mark options={fill=$col_ipopt}");
t3 = trajectory_plot(p3, mark="*", legendentry="SNOPT", style="very thick, color=$col_snopt, mark options={fill=$col_snopt}");


goal = ([x0[1], xf[1]],
        [x0[2], xf[2]])
z = ["start","end"]
g = PGF.Plots.Scatter(goal[1], goal[2], z,
    scatterClasses="{start={yellow, mark=*, yellow, scale=2},
        end={mark=square*, red, scale=2}}",);

a = Axis([p; t3; t2; t1; g],
    xmin=-0.1, ymin=-1, xmax=1.5, ymax=1,
    axisEqualImage=true,
    legendPos="north west",
    hideAxis=true)

# Save to tikz format
# NOTE: To fix the problem with the legend for the start and goal points, replace \addplot+ with \addplot in the tikz file
save(joinpath(paper,"3obs_traj.tikz"), a, include_preamble=false)
