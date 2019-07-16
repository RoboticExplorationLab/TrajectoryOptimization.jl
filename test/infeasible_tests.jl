T = Float64

## Pendulum
xf = Problems.pendulum_problem.xf
x0 = Problems.pendulum_problem.x0
N = Problems.pendulum_problem.N
n = Problems.pendulum_problem.model.n; m = Problems.pendulum_problem.model.m

# options
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,constraint_tolerance=1.0e-5,
    cost_tolerance=1.0e-5,cost_tolerance_intermediate=1.0e-5,opts_uncon=opts_ilqr,iterations=30,
    penalty_scaling=10.0)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=1.0,resolve_feasible_problem=false)
opts_altro_resolve = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=1.0,resolve_feasible_problem=true)

X0 = line_trajectory(x0,xf,N)

# unconstrained infeasible solve
prob = update_problem(copy(Problems.pendulum_problem))
copyto!(prob.X,X0)
solve!(prob,opts_altro)
@test norm(prob.X[end] - xf) < 1.0e-3

prob_resolve = update_problem(copy(Problems.pendulum_problem))
copyto!(prob_resolve.X,X0)
solve!(prob_resolve,opts_altro_resolve)
@test norm(prob_resolve.X[end] - xf) < 1.0e-3

@test norm(prob.X[end] - prob_resolve.X[end]) < 1.0e-5

# constrained infeasible solve
u_bnd = 3.
bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd,trim=true)
goal = goal_constraint(xf)
constraints = Constraints([bnd],N)

prob = update_problem(copy(Problems.pendulum_problem),constraints=copy(constraints))
prob.constraints[N] += goal
copyto!(prob.X,X0)
solve!(prob,opts_altro)

@test max_violation(prob) < opts_al.constraint_tolerance

prob_resolve = update_problem(copy(Problems.pendulum_problem),constraints=copy(constraints))
prob_resolve.constraints[N] += goal

copyto!(prob_resolve.X,X0)
solve!(prob_resolve,opts_altro_resolve)
@test max_violation(prob_resolve) < opts_al.constraint_tolerance

@test norm(prob.X[end] - prob_resolve.X[end]) < 1.0e-5

## Quadrotor in Maze
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,iterations=300,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    iterations=40,cost_tolerance=1.0e-5,cost_tolerance_intermediate=1.0e-4,constraint_tolerance=1.0e-3,penalty_scaling=10.,penalty_initial=1.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,resolve_feasible_problem=false,opts_al=opts_al,R_inf=0.001);
opts_altro_resolve = ALTROSolverOptions{T}(verbose=verbose,resolve_feasible_problem=false,opts_al=opts_al,R_inf=0.001);

prob = copy(Problems.quadrotor_maze_problem)
solve!(prob,opts_altro)
@test max_violation(prob) < opts_al.constraint_tolerance

prob_resolve = copy(Problems.quadrotor_maze_problem)

solve!(prob_resolve,opts_altro_resolve)
@test max_violation(prob_resolve) < opts_al.constraint_tolerance
@test norm(prob.X[end] - prob_resolve.X[end]) < 1.0e-3
