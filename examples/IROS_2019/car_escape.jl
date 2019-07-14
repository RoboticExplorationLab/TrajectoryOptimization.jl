using BenchmarkTools, SNOPT7

# Car escape
T = Float64

# options
max_con_viol = 1.0e-6
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=10.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,feasibility_tolerance=max_con_viol, solve_type=:feasible)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-3,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-5);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    opts=Dict(:print_level=>3,
        :tol=>max_con_viol,
        :constr_viol_tol=>max_con_viol))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    opts=Dict(:Major_print_level=>0,
        :Minor_print_level=>0,
        :Major_optimality_tolerance=>max_con_viol,
        :Major_feasibility_tolerance=>max_con_viol,
        :Minor_feasibility_tolerance=>max_con_viol))

x0 = Problems.car_escape_problem.x0
xf = Problems.car_escape_problem.xf

# ALTRO
prob_altro = copy(Problems.car_escape_problem)
@time p1, s1 = solve(prob_altro, opts_altro)
# b1 = @benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p1)
Problems.plot_escape(p1.X,x0,xf)


# DIRCOL w/ Ipopt
prob_ipopt = update_problem(copy(Problems.car_escape_problem),model=Dynamics.car_model) # get continuous time model
p2, s2 = solve(prob_ipopt, opts_ipopt)
# b2 = @benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation(p2)
Problems.plot_escape(p2.X,x0,xf)

# DIRCOL w/ SNOPT
prob_snopt = update_problem(copy(Problems.car_escape_problem),model=Dynamics.car_model) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
# @benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation(p3)
Problems.plot_escape(p3.X,x0,xf)

# AL-iLQR
prob_altro = copy(Problems.car_escape_problem)
prob_altro.X .*= NaN
opts_altro.projected_newton = false
@time p4, s4 = solve(prob_altro, opts_altro)
# b1 = @benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p4)
Problems.plot_escape(p4.X,x0,xf)
x = [x[1] for x in p4.X]
y = [x[2] for x in p4.X]
PGF.Plots.Linear(x,y);


###############################################
#            Create PGF Plot                  #
###############################################

# Plot the walls
color_wall = "gray"
style = "color=$color_wall, fill=$color_wall"
function trajectory_plot(prob::Problem; kwargs...)
    x = [x[1] for x in prob.X]
    y = [x[2] for x in prob.X]
    PGF.Plots.Linear(x,y; kwargs...)
end
p = [PGF.Plots.Circle(circle..., style=style) for circle in circles]

# Plot the trajectories
t0 = PGF.Plots.Linear(Problems.X0_escape[1,:],Problems.X0_escape[2,:],
    legendentry="initial guess",
    mark="none",
    style="color=black, dashed, thick")
t1 = trajectory_plot(p1, mark="none", legendentry="ALTRO", style="very thick, color=orange!80!black")
t3 = trajectory_plot(p3, mark="none", legendentry="SNOPT", style="very thick, color=green, dashed");
t4 = trajectory_plot(p4, mark="none", legendentry="AL-iLQR", style="very thick, color=cyan");

# Plot the initial and final points
goal = ([x0[1], xf[1]],
        [x0[2], xf[2]])
z = ["start","end"]
g = PGF.Plots.Scatter(goal[1], goal[2], z,
    scatterClasses="{start={green, mark=*, green, scale=2},
        end={mark=square*, red, scale=2}}",
    legendentry=["start", "end"])

# Plot the whole thing
a = Axis([p; t0; t1; t4; t3; g],
    xmin=-1, ymin=-1, xmax=11, ymax=8,
    axisEqualImage=true,
    legendPos="north west",
    hideAxis=true)

# Save to tikz format
# NOTE: To fix the problem with the legend for the start and goal points, replace \addplot+ with \addplot in the tikz file
save(joinpath(paper,"escape_traj.tikz"), a, include_preamble=false)
