using BenchmarkTools, Plots, SNOPT7

# options
T = Float64
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    iterations=30,penalty_scaling=10.0)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_minimum_time=10.0,
    dt_max=0.2,dt_min=1.0e-3)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:print_level=>3,:tol=>1.0e-3,:constr_viol_tol=>1.0e-3))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,:Minor_print_level=>0,:Major_optimality_tolerance=>1.0e-3,
        :Major_feasibility_tolerance=>1.0e-3, :Minor_feasibility_tolerance=>1.0e-3))

opts_mt_ipopt = DIRCOLSolverMTOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:print_level=>3,:tol=>1.0e-3,:constr_viol_tol=>1.0e-3),R_min_time=10.0,h_max=0.2,h_min=1.0e-3)

opts_mt_snopt = DIRCOLSolverMTOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,:Minor_print_level=>0,:Major_optimality_tolerance=>1.0e-3,
        :Major_feasibility_tolerance=>1.0e-3, :Minor_feasibility_tolerance=>1.0e-3),R_min_time=10.0,h_max=0.2,h_min=1.0e-3)

## Solver comparison

# ALTRO w/o Newton
prob_altro = copy(Problems.box_parallel_park_problem)
p1, s1 = solve(prob_altro, opts_altro)
# @benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p1)
X1 = to_array(p1.X)
plot(X1[1,:],X1[2,:],title="Parallel Park (box) - (ALTRO)")
plot(p1.U,title="Parallel Park (box) - (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.box_parallel_park_problem)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.car_model) # get continuous time model
p2, s2 = solve(prob_ipopt, opts_ipopt)
# @benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation(p2)
X2 = to_array(p2.X)
plot(X2[1,:],X2[2,:],title="Parallel Park (box) - (Ipopt)")
plot(p2.U,title="Parallel Park (box) - (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.box_parallel_park_problem)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.car_model) # get continuous time model
p3, s3 = solve(prob_snopt, opts_snopt)
# @benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation(p3)
X3 = to_array(p3.X)
plot(X3[1,:],X3[2,:],title="Parallel Park (box) - (SNOPT)")
plot(p3.U,title="Parallel Park (box) - (SNOPT)")

## Minimum Time

# ALTRO w/o Newton
prob_mt_altro = update_problem(copy(Problems.box_parallel_park_problem),tf=0.) # make minimum time problem by setting tf = 0
initial_controls!(prob_mt_altro,p1.U)
p4, s4 = solve(prob_mt_altro,opts_altro)
# @benchmark p4, s4 = solve($prob_mt_altro, $opts_altro)
max_violation(p4)
X4 = to_array(p4.X)
plot(X4[1,:],X4[2,:],title="Parallel Park Min. Time (box) - (ALTRO)")
plot(p4.U,title="Parallel Park (box) - (ALTRO)")

# DIRCOL w/ Ipopt
prob_mt_ipopt = copy(Problems.box_parallel_park_problem)
initial_controls!(prob_mt_ipopt,p2.U)
rollout!(prob_mt_ipopt)
prob_mt_ipopt = update_problem(prob_mt_ipopt,model=Dynamics.car_model) # get continuous time model
p5, s5 = solve(prob_mt_ipopt, opts_mt_ipopt)
# @benchmark p5, s5 = solve($prob_mt_ipopt, $opts_mt_ipopt)
max_violation(p5)
X5 = to_array(p5.X)
plot(X5[1,:],X5[2,:],title="Parallel Park Min. Time (box) - (Ipopt)")
plot(p5.U,title="Parallel Park (box) - (Ipopt)")

# DIRCOL w/ SNOPT
prob_mt_snopt = copy(Problems.box_parallel_park_problem)
initial_controls!(prob_mt_snopt,p3.U)
rollout!(prob_mt_snopt)
prob_mt_snopt = update_problem(prob_mt_snopt,model=Dynamics.car_model) # get continuous time model
p6, s6 = solve(prob_mt_snopt, opts_mt_snopt)
# @benchmark p6, s6 = solve($prob_mt_snopt, $opts_mt_snopt)
max_violation(p6)
X6 = to_array(p6.X)
plot(X6[1,:],X6[2,:],title="Parallel Park Min. Time (box) - (SNOPT)")
plot(p6.U,title="Parallel Park (box) - (SNOPT)")
