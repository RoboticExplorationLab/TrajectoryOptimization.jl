# options
T = Float64
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,
    iterations=30,penalty_scaling=10.0)

opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=10.0,
    dt_max=0.2,dt_min=1.0e-3)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:tol=>1.0e-3,:constr_viol_tol=>1.0e-3))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,:Minor_print_level=>0,:Major_optimality_tolerance=>1.0e-3,
        :Major_feasibility_tolerance=>1.0e-3, :Minor_feasibility_tolerance=>1.0e-3))

# ALTRO w/o Newton
prob_altro = copy(Problems.box_parallel_park_problem)
p1, s1 = solve(prob_altro, opts_altro)
# @btime p1, s1 = solve($prob_altro, $opts_altro)

# DIRCOL w/ Ipopt
prob_ipopt = update_problem(copy(Problems.box_parallel_park_problem),model=Dynamics.car_model) # get continuous time model
p2, s2 = solve(prob_ipopt, opts_ipopt)
# @btime p2, s2 = solve($prob_ipopt, $opts_ipopt)

# DIRCOL w/ SNOPT
prob_snopt = update_problem(copy(Problems.box_parallel_park_problem),model=Dynamics.car_model) # get continuous time model
p3, s3 = solve(prob_snopt, opts_snopt)
# @btime p3, s3 = solve($prob_snopt, $opts_snopt)

## Minimum Time
prob_mt = copy(Problems.box_parallel_park_min_time_problem)
initial_controls!(prob_mt,prob.U)
solve!(prob_mt,opts_altro)
