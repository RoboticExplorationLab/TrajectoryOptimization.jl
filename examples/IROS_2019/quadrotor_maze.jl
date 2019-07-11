# Quadrotor in Maze
T = Float64

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,iterations=300,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    iterations=40,cost_tolerance=1.0e-5,cost_tolerance_intermediate=1.0e-4,constraint_tolerance=1.0e-3,penalty_scaling=10.,penalty_initial=1.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=0.001)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:tol=>1.0e-3,:constr_viol_tol=>1.0e-3))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,:Minor_print_level=>0,:Major_optimality_tolerance=>1.0e-3,
        :Major_feasibility_tolerance=>1.0e-3, :Minor_feasibility_tolerance=>1.0e-3))


# ALTRO w/o Newton
prob_altro = copy(Problems.quadrotor_maze_problem)
p1, s1 = solve(prob_altro, opts_altro)
# @btime p1, s1 = solve($prob_altro, $opts_altro)

# DIRCOL w/ Ipopt
prob_ipopt = update_problem(copy(Problems.quadrotor_maze_problem),model=Dynamics.quadrotor_model) # get continuous time model
p2, s2 = solve(prob_ipopt, opts_ipopt)
# @btime p2, s2 = solve($prob_ipopt, $opts_ipopt)

# DIRCOL w/ SNOPT
prob_snopt = update_problem(copy(Problems.quadrotor_maze_problem),model=Dynamics.quadrotor_model) # get continuous time model
p3, s3 = solve(prob_snopt, opts_snopt)
# @btime p3, s3 = solve($prob_snopt, $opts_snopt)
