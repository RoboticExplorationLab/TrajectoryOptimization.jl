using BenchmarkTools, Plots, SNOPT7

# Car escape
T = Float64

# options
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,cost_tolerance_intermediate=1.0e-2,constraint_tolerance=1.0e-3,penalty_scaling=50.,penalty_initial=10.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=1.0e-3);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:tol=>1.0e-3,:constr_viol_tol=>1.0e-3))
opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_optimality_tolerance=>1.0e-3,
        :Major_feasibility_tolerance=>1.0e-3, :Minor_feasibility_tolerance=>1.0e-3))


# ALTRO w/o Newton
prob_altro = copy(Problems.car_escape_problem)
solve!(prob_altro, opts_altro)

# DIRCOL w/ Ipopt
prob_ipopt = update_problem(copy(Problems.car_escape_problem),model=Dynamics.car_model) # get continuous time model
solve!(prob_ipopt, opts_ipopt)

# DIRCOL w/ SNOPT
prob_snopt = update_problem(copy(Problems.car_escape_problem),model=Dynamics.car_model) # get continuous time model
solve!(prob_snopt, opts_snopt)


# Problems.plot_escape()
