using BenchmarkTools, SNOPT7, Plots
# Quadrotor in Maze
T = Float64

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,iterations=300,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    iterations=40,cost_tolerance=1.0e-5,cost_tolerance_intermediate=1.0e-4,constraint_tolerance=1.0e-3,penalty_scaling=10.,penalty_initial=1.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=0.001)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:print_level=>3,:tol=>1.0e-3,:constr_viol_tol=>1.0e-3))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,:Minor_print_level=>0,:Major_optimality_tolerance=>1.0e-3,
        :Major_feasibility_tolerance=>1.0e-3, :Minor_feasibility_tolerance=>1.0e-3))


# ALTRO w/o Newton
prob_altro = copy(Problems.quadrotor_wall_problem)
p1, s1 = solve(prob_altro, opts_altro)
# @benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p1)
X1 = to_array(p1.X)
plot(X1[1:3,:]',title="Quadrotor position (ALTRO)")
plot(p1.U,title="Quadrotor control (ALTRO)")


# DIRCOL w/ Ipopt
prob_ipopt = update_problem(copy(Problems.quadrotor_wall_problem),model=Dynamics.quadrotor_model) # get continuous time model
p2, s2 = solve(prob_ipopt, opts_ipopt)
# @benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation(p2)
X2 = to_array(p2.X)
plot(X2[1:3,:]',title="Quadrotor position (Ipopt)")
plot(p2.U,title="Quadrotor control (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = update_problem(copy(Problems.quadrotor_wall_problem),model=Dynamics.quadrotor_model) # get continuous time model
p3, s3 = solve(prob_snopt, opts_snopt)
# @benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation(p3)
X3 = to_array(p3.X)
plot(X3[1:3,:]',title="Quadrotor position (SNOPT)")
plot(p3.U,title="Quadrotor control (SNOPT)")
