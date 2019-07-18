using Random, BenchmarkTools, SNOPT7
# using PGFPlots
# const PGF = PGFPlots

# Car escape
T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    cost_tolerance_intermediate=1.0e-2,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=10.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol,
    solve_type=:feasible)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-1,
    resolve_feasible_problem=false,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:Ipopt,
    feasibility_tolerance=max_con_viol)

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=:SNOPT7,
    feasibility_tolerance=max_con_viol)

x0 = Problems.car_escape_problem.x0
xf = Problems.car_escape_problem.xf

# ALTRO

prob_altro = copy(Problems.car_escape_problem)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation_direct(p1)
Problems.plot_escape(p1.X,x0,xf)

# DIRCOL w/ Ipopt
prob_ipopt = update_problem(copy(Problems.car_escape_problem),model=Dynamics.car_model) # get continuous time model
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation_direct(p2)
Problems.plot_escape(p2.X,x0,xf)

# DIRCOL w/ SNOPT
prob_snopt = update_problem(copy(Problems.car_escape_problem),model=Dynamics.car_model) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
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

# Max constraint plot
t_pn = s1.stats[:time_al]
t_span_al = range(0,stop=s1.stats[:time_al],length=s1.solver_al.stats[:iterations])
t_span_pn = range(t_pn,stop=s1.stats[:time],length=s1.solver_pn.stats[:iterations]+1)
t_span = [t_span_al;t_span_pn[2:end]]
c_span = [s1.solver_al.stats[:c_max]...,s1.solver_pn.stats[:c_max]...]

s_range = [1,(2:1:length(s3.stats[:iter_time])-1)...,length(s3.stats[:iter_time])]
i_range = [1,(2:1:length(s2.stats[:iter_time])-1)...,length(s2.stats[:iter_time])]
p = plot(t_pn*ones(100),range(1.0e-9,stop=1.0,length=100),color=:red,linestyle=:dash,label="Projected Newton",width=2)
p = plot!(s3.stats[:iter_time][s_range],s3.stats[:c_max][s_range],marker=:circle,yscale=:log10,ylim=[1.0e-9,1.0],color=:green,label="SNOPT")
p = plot!(s2.stats[:iter_time][i_range],s2.stats[:c_max][i_range],marker=:circle,yscale=:log10,ylim=[1.0e-9,1.0],color=:blue,label="Ipopt")
p = plot!(t_span,c_span,title="Car Escape c_max",xlabel="time (s)",marker=:circle,color=:orange,width=2,yscale=:log10,ylim=[1.0e-9,1.0],label="ALTRO")

savefig(p,joinpath(pwd(),"examples/IROS_2019/car_escape_c_max.png"))

#PGFPlots version
w1 = PGF.Plots.Linear(s3.stats[:iter_time][s_range],s3.stats[:c_max][s_range],mark="none", legendentry="DIRCOL-S",style="very thick, color=green")
w2 = PGF.Plots.Linear(s2.stats[:iter_time][i_range],s2.stats[:c_max][i_range], mark="none",legendentry="DIRCOL-I",style="very thick, color=cyan")
w3 = PGF.Plots.Linear(t_span,c_span, mark="none", legendentry="ALTRO",style="very thick, color=orange!80!black")
w4 = PGF.Plots.Linear(t_pn*ones(100),range(1.0e-9,stop=1.0,length=100),legendentry="Projected Newton",mark="none",style="very thick, color=red, dashed")

a = Axis([w4;w1;w2;w3],
    xmin=0., ymin=1e-9, xmax=45, ymax=1.0,
    legendPos="north east",
    ymode="log",
    hideAxis=false,
    xlabel="time (s)")

paper = "/home/taylor/Documents/research/ALTRO_paper/images"
save(joinpath(paper,"escape_c_max.tikz"), a, include_preamble=false)
