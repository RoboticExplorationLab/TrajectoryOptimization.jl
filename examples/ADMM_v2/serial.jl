using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using LinearAlgebra
using ForwardDiff
using Combinatorics
using TrajectoryOptimization
const TO = TrajectoryOptimization

include("problem.jl")
include("methods.jl")

# Solver options
verbose=false

opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=500)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=0.001,
    cost_tolerance_intermediate=1.0e-4,
    iterations=30,
    penalty_scaling=2.0,
    penalty_initial=10.)


num_lift = 4
quat = true
r0_load = [0, 0, 0.25]
scenario = :p2p
prob_load = gen_prob(:load, quad_params, load_params, r0_load,
    num_lift=num_lift, quat=quat, scenario=scenario)
prob_lift = [gen_prob(i, quad_params, load_params, r0_load,
    num_lift=num_lift, quat=quat, scenario=scenario) for i = 1:num_lift]

# @time plift_al, pload_al, slift_al, sload_al = solve_admm_1slack(prob_lift,prob_load,:parallel,opts_al)
prob_lift, prob_load = trim_conditions(num_lift, r0_load, quad_params, load_params, quat, opts_al)
@time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift, prob_load, quad_params,
    load_params, :parallel, opts_al, max_iters=3)
@btime solve_admm($prob_lift, $prob_load, $quad_params,
    $load_params, :parallel, $opts_al, max_iters=3)

visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al])
#=
x: [-0.5, 0.5]
=#

visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al])

prob_load2 = gen_prob_all(quad_params, load_params, agent=:load)
prob_lift2 = [gen_prob_all(quad_params, load_params, agent=i) for i = 1:num_lift]
@time plift_al, pload_al, slift_al, sload_al = solve_admm_1slack(prob_lift2,prob_load2,quad_params, load_params,:parallel,opts_al)

include("visualization.jl")
vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al])

# for i = 1:num_lift
#     initial_controls!(prob_lift[i],[plift_al[i].U[end] for k = 1:prob_load.N])
#     prob_lift[i].x0[4:7] = plift_al[i].X[end][4:7]
#     rollout!(prob_lift[i])
#     println(plift_al[i].X[end][4:7])
#     println(plift_al[i].U[end])
# end
# initial_controls!(prob_load,[pload_al.U[end] for k = 1:prob_load.N])
# println(pload_al.U[end])
#
# rollout!(prob_load)
#
# visualize_quadrotor_lift_system(vis, [[prob_load]; prob_lift])

#
plot(plift_al[2].U,1:4)
