using BenchmarkTools, Plots, SNOPT7

T = Float64

# options
max_con_viol = 1.0e-6
verbose=true

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,cost_tolerance_intermediate=1.0e-3,constraint_tolerance=max_con_viol,penalty_scaling=100.,penalty_initial=1.)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,projected_newton=true,projected_newton_tolerance=1.0e-6,opts_al=opts_al,opts_pn=opts_pn);

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt, opts=Dict(:print_level=>3,:tol=>max_con_viol,:constr_viol_tol=>max_con_viol))

opts_snopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:SNOPT7, opts=Dict(:Major_print_level=>0,:Minor_print_level=>0,:Major_optimality_tolerance=>max_con_viol,
        :Major_feasibility_tolerance=>max_con_viol, :Minor_feasibility_tolerance=>max_con_viol))

xf = Problems.pendulum_problem.xf
N = Problems.pendulum_problem.N
goal = goal_constraint(xf)

# ALTRO w/o Newton
prob_altro = copy(Problems.pendulum_problem)
prob_altro.constraints[N] += goal
@time p1, s1 = solve(prob_altro, opts_altro)
# @benchmark p1, s1 = solve($prob_altro, $opts_altro)
max_violation(p1)
plot(p1.X,title="Pendulum state (ALTRO)")
plot(p1.U,title="Pendulum control (ALTRO)")

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.pendulum_problem)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.pendulum_model) # get continuous time model
prob_ipopt.constraints[N] += goal
@time p2, s2 = solve(prob_ipopt, opts_ipopt)
# @benchmark p2, s2 = solve($prob_ipopt, $opts_ipopt)
max_violation(p2)
plot(p2.X,title="Pendulum state (Ipopt)")
plot(p2.U,title="Pendulum control (Ipopt)")

# DIRCOL w/ SNOPT
prob_snopt = copy(Problems.pendulum_problem)
rollout!(prob_snopt)
prob_snopt = update_problem(prob_snopt,model=Dynamics.pendulum_model) # get continuous time model
prob_snopt.constraints[N] += goal
@time p3, s3 = solve(prob_snopt, opts_snopt)
# @benchmark p3, s3 = solve($prob_snopt, $opts_snopt)
max_violation(p3)
plot(p3.X,title="Pendulum state (SNOPT)")
plot(p3.U,title="Pendulum control (SNOPT)")
