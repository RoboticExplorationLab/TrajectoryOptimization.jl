# options
T = Float64
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,
    iterations=30,penalty_scaling=10.0)

opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=10.0,
    dt_max=0.2,dt_min=1.0e-3)

# solve
prob = copy(Problems.box_parallel_park_problem)
solve!(prob,opts_altro)
tt = total_time(prob)

prob_mt = copy(Problems.box_parallel_park_min_time_problem)
initial_controls!(prob_mt,prob.U)
solve!(prob_mt,opts_altro)
tt_mt = total_time(prob_mt)

@test tt_mt < 0.75*tt
@test tt_mt < 2.1

@test norm(prob_mt.X[end] - xf,Inf) < 1e-3
@test max_violation(prob_mt) < opts_al.constraint_tolerance
