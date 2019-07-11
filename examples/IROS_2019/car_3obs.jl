using BenchmarkTools, Plots, SNOPT7

# Car escape
T = Float64

# options
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,cost_tolerance_intermediate=1.0e-2,constraint_tolerance=1.0e-3,penalty_scaling=50.,penalty_initial=10.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=1.0e-3);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:print_level=>3,:tol=>1.0e-3,:constr_viol_tol=>1.0e-3))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,:Minor_print_level=>0,:Major_optimality_tolerance=>1.0e-3,
        :Major_feasibility_tolerance=>1.0e-3, :Minor_feasibility_tolerance=>1.0e-3))

x0 = Problems.car_3obs_problem.x0
xf = Problems.car_3obs_problem.xf

# ALTRO w/o Newton
prob_altro = copy(Problems.car_3obs_problem)
p1, s1 = solve(prob_altro, opts_altro)
# @benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p1)

Problems.plot_car_3obj(p1.X,x0,xf)

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.car_3obs_problem)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.car_model) # get continuous time model)
p2, s2 = solve(prob_ipopt, opts_ipopt)
# @benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation(p2)
Problems.plot_car_3obj(p2.X,x0,xf)


# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.car_3obs_problem)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.car_model) # get continuous time model
p3, s3 = solve(prob_snopt, opts_snopt)
# @benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation(p3)

Problems.plot_car_3obj(p3.X,x0,xf)
