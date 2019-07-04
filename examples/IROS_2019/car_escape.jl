# Car escape
T = Float64

# options
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-4,cost_tolerance_intermediate=1.0e-2,constraint_tolerance=1.0e-3,penalty_scaling=50.,penalty_initial=10.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=1.0e-3);

prob = copy(Problems.car_escape_problem)

solve!(prob, opts_altro)
