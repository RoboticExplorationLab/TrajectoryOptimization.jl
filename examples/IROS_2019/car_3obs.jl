using BenchmarkTools, Plots, SNOPT7

# Car escape
T = Float64

# options
max_con_viol = 1.0e-8
verbose=true

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,cost_tolerance_intermediate=1.0e-2,constraint_tolerance=max_con_viol,
    penalty_scaling=50.,penalty_initial=10.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,opts_pn=opts_pn,
    projected_newton=true,projected_newton_tolerance=1.0e-3);

# opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt,
#     opts=Dict(:print_level=>3,:tol=>max_con_viol,:constr_viol_tol=>max_con_viol))
#
# opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,
#     :Minor_print_level=>0,:Major_optimality_tolerance=>max_con_viol,
#     :Major_feasibility_tolerance=>max_con_viol, :Minor_feasibility_tolerance=>max_con_viol))

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt,
    opts=Dict(:tol=>max_con_viol,:constr_viol_tol=>max_con_viol))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_optimality_tolerance=>max_con_viol,
    :Major_feasibility_tolerance=>max_con_viol, :Minor_feasibility_tolerance=>max_con_viol))


x0 = Problems.car_3obs_problem.x0
xf = Problems.car_3obs_problem.xf

# ALTRO w/ Newton
prob_altro = copy(Problems.car_3obs_problem)
@time p1, s1 = solve(prob_altro, opts_altro)
@benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p1)
max_violation_dynamics(p1)

Problems.plot_car_3obj(p1.X,x0,xf)

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.car_3obs_problem)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.car_model) # get continuous time model)
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
@benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation(p2)
max_violation_dynamics(p2,s2)
Problems.plot_car_3obj(p2.X,x0,xf)

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.car_3obs_problem)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.car_model) # get continuous time model
@time p3, s3 = solve(prob_snopt, opts_snopt)
@benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation(p3)
max_violation_dynamics(p3,s3)

Problems.plot_car_3obj(p3.X,x0,xf)
