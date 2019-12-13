
T = Float64

# options
max_con_viol = 1.0e-8
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,
    cost_tolerance_intermediate=1.0e-3,
    constraint_tolerance=max_con_viol,
    penalty_scaling=50.,
    penalty_initial=1.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,
    feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    opts_pn=opts_pn,
    projected_newton=true,
    projected_newton_tolerance=1.0e-3);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,
    nlp=Ipopt.Optimizer(),
    feasibility_tolerance=max_con_viol)

# opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,
#     nlp=SNOPT7.Optimizer(),
#     feasibility_tolerance=max_con_viol)

prob_altro = copy(Problems.cartpole)
@time p0, s0 = solve(prob_altro, opts_al)
max_violation(p0)
@btime solve($prob_altro, $opts_al)

# ALTRO w/ Newton
prob_altro = copy(Problems.cartpole)
@time p1, s1 = solve(prob_altro, opts_altro)
@btime p1, s1 = solve($prob_altro, $opts_altro)
max_violation_direct(p1)
plot(p1.X,title="Cartpole state (ALTRO)")
plot(p1.U,title="Cartpole control (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.cartpole)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.cartpole) # get continuous time model
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@btime p2, s2 = solve($prob_ipopt, $opts_ipopt)
push!(s2.stats[:iter_time],s2.stats[:time])
push!(s2.stats[:c_max], max_violation_direct(p2))
max_violation_direct(p2)
plot(p2.X,title="Cartpole state (Ipopt)")
plot(p2.U,title="Cartpole control (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.cartpole)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.cartpole) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation_direct(p3)
s3.stats[:c_max][end]
plot(p3.X,title="Cartpole state (SNOPT)")
plot(p3.U,title="Cartpole control (SNOPT)")

# AL-iLQR
prob_alilqr = copy(Problems.cartpole)
opts_al.constraint_tolerance = max_con_viol
@time p4, s4 = solve(prob_alilqr, opts_al)
@benchmark p4, s4 = solve($prob_alilqr, $opts_al)
max_violation_direct(p4)
plot(p4.X,title="Cartpole state (ALTRO)")
plot(p4.U,title="Cartpole control (ALTRO)")

# Convergence plot
t_pn = s1.stats[:time_al]
t_span_al = range(0,stop=s1.stats[:time_al],length=s1.solver_al.stats[:iterations])
t_span_pn = range(t_pn,stop=s1.stats[:time],length=s1.solver_pn.stats[:iterations]+1)
t_span = [t_span_al;t_span_pn[2:end]]
c_span = [s1.solver_al.stats[:c_max]...,s1.solver_pn.stats[:c_max]...]

al_iter = s1.solver_al.stats[:iterations]
t_span2 = [t_span[1:al_iter]; t_pn .+ collect(range(0,stop=s4.stats[:time]-t_pn, length=s4.stats[:iterations]-al_iter))]
c_span2 = s4.stats[:c_max]


s_range = [1,(2:4:length(s3.stats[:iter_time])-1)...,length(s3.stats[:iter_time])]
i_range = [1,(2:4:length(s2.stats[:iter_time])-1)...,length(s2.stats[:iter_time])]
# p = plot(t_pn*ones(100),range(1.0e-10,stop=10.0,length=100),color=:red,linestyle=:dash,label="Projected Newton",width=2)
# p = plot!(s3.stats[:iter_time][s_range],s3.stats[:c_max][s_range],marker=:circle,yscale=:log10,ylim=[1.0e-10,10.0],color=:green,label="SNOPT")
# p = plot!(s2.stats[:iter_time][i_range],s2.stats[:c_max][i_range],marker=:circle,yscale=:log10,ylim=[1.0e-10,10.0],color=:blue,label="Ipopt")
# p = plot!(t_span,c_span,title="Cartpole c_max",xlabel="time (s)",marker=:circle,color=:orange,width=2,yscale=:log10,ylim=[1.0e-10,10.0],label="ALTRO")

###############################################
#            Create PGF Plot                  #
###############################################
include("vars.jl")
lwidth = 1.5
w1 = PGF.Plots.Linear(s3.stats[:iter_time][s_range],s3.stats[:c_max][s_range],mark="none", legendentry="SNOPT",style="line width=$lwidth, color=$col_snopt");
w2 = PGF.Plots.Linear(s2.stats[:iter_time][i_range],s2.stats[:c_max][i_range], mark="none",legendentry="Ipopt",style="line width=$lwidth, color=$col_ipopt");
w3 = PGF.Plots.Linear(t_span, c_span,  mark="none", legendentry="ALTRO",  style="line width=$lwidth, color=$col_altro");
w4 = PGF.Plots.Linear(t_span2,c_span2, mark="none", legendentry="AL-iLQR",style="line width=$lwidth, color=$col_alilqr");
w5 = PGF.Plots.Linear(t_pn*ones(100),range(1.0e-9,stop=1e5,length=100),legendentry="Projected Newton",mark="none",style="thick, color=red, dashed");

a = Axis([w1;w2;w4;w3;w5],
    xmin=0., ymin=1e-9, xmax=s3.stats[:iter_time][end], ymax=s3.stats[:c_max][1],
    legendPos="north east",
    ymode="log",
    hideAxis=false,
    xlabel="time (s)",
    ylabel="maximum constraint violation",
    style="grid=both")

save(joinpath(paper,"cartpole_c_max.tikz"), a, include_preamble=false)
