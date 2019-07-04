# Quadrotor in Maze
T = Float64

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,iterations=300,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    iterations=40,cost_tolerance=1.0e-5,cost_tolerance_intermediate=1.0e-4,constraint_tolerance=1.0e-3,penalty_scaling=10.,penalty_initial=1.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=0.001);

prob = copy(Problems.quadrotor_maze_problem)
solve!(prob,opts_altro)

@test max_violation(prob) < opts_al.constraint_tolerance
