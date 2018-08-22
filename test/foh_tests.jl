using TrajectoryOptimization
using Plots

dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = false
opts.c1 = 1e-4
opts.c2 = 20.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-5
opts.iterations_outerloop = 250
opts.iterations = 100

obj_uncon = TrajectoryOptimization.Dynamics.pendulum![2]

###
solver_foh = TrajectoryOptimization.Solver(Dynamics.pendulum![1], obj_uncon, dt=dt,integration=:rk3_foh, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(Dynamics.pendulum![1], obj_uncon, dt=dt,integration=:rk3, opts=opts)

U = ones(solver_foh.model.m, solver_foh.N)

sol_zoh = solve(solver_zoh,U)
sol_foh = solve(solver_foh,U)

results_foh = TrajectoryOptimization.UnconstrainedResults(solver_foh.model.n,solver_foh.model.m,solver_foh.N)
results_foh.X .= sol_foh.X
results_foh.U .= sol_zoh.U
backwardpass_foh!(results_foh,solver_foh)
rollout!(results_foh,solver_foh,0.01)
cost(solver_foh,results_foh.X_,results_foh.U_)


x = 4
# results_zoh = TrajectoryOptimization.UnconstrainedResults(solver_zoh.model.n,solver_zoh.model.m,solver_zoh.N)
#
# results_foh.U[:,:] .= U
# results_zoh.U[:,:] .= U
#
# TrajectoryOptimization.rollout!(results_foh,solver_foh)
# TrajectoryOptimization.rollout!(results_zoh,solver_zoh)
#
# TrajectoryOptimization.cost(solver_foh,results_foh.X,results_foh.U)
# TrajectoryOptimization.cost(solver_zoh,results_zoh.X,results_zoh.U)
#
# plot(results_foh.X')
# plot!(results_zoh.X')
#
# backwardpass_foh!(results_foh,solver_foh)
#
# calc_jacobians(results_zoh,solver_zoh)
# backwardpass!(results_zoh,solver_zoh)
#
# rollout!(results_foh,solver_foh,0.5)
# rollout!(results_zoh,solver_zoh,0.5)
#
# cost(solver_foh,results_foh.X_,results_foh.U_)
# cost(solver_zoh,results_zoh.X_,results_zoh.U_)
#
# results_foh.X .= results_foh.X_
# results_foh.U .= results_foh.U_
#
# results_zoh.X .= results_zoh.X_
# results_zoh.U .= results_zoh.U_
#
# backwardpass_foh!(results_foh,solver_foh)
#
# calc_jacobians(results_zoh,solver_zoh)
# backwardpass!(results_zoh,solver_zoh)
#
# rollout!(results_foh,solver_foh,0.5)
# rollout!(results_zoh,solver_zoh,0.5)
#
# cost(solver_foh,results_foh.X_,results_foh.U_)
# cost(solver_zoh,results_zoh.X_,results_zoh.U_)
#
