using BenchmarkTools, Plots, SNOPT7

T = Float64

# options
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,cost_tolerance_intermediate=1.0e-3,constraint_tolerance=1.0e-3,penalty_scaling=10.,penalty_initial=1.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:print_level=>3,:tol=>1.0e-3,:constr_viol_tol=>1.0e-3))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,:Minor_print_level=>0,:Major_optimality_tolerance=>1.0e-3,
        :Major_feasibility_tolerance=>1.0e-3, :Minor_feasibility_tolerance=>1.0e-3))


# ALTRO w/o Newton
prob_altro = copy(Problems.cartpole_problem)
p1, s1 = solve(prob_altro, opts_altro)
# @benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p1)
plot(p1.X,title="Cartpole state (ALTRO)")
plot(p1.U,title="Cartpole control (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.cartpole_problem)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.cartpole_model) # get continuous time model
p2, s2 = solve(prob_ipopt, opts_ipopt)
# @benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation(p2)
plot(p2.X,title="Cartpole state (Ipopt)")
plot(p2.U,title="Cartpole control (Ipopt)")


# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.cartpole_problem)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.cartpole_model) # get continuous time model
p3, s3 = solve(prob_snopt, opts_snopt)
# @benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation(p3)
plot(p3.X,title="Cartpole state (SNOPT)")
plot(p3.U,title="Cartpole control (SNOPT)")
