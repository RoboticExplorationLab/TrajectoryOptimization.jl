using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using LinearAlgebra
using ForwardDiff
const TO = TrajectoryOptimization

include("problem.jl")
include("methods.jl")

# Solver options
verbose=false

opts_ilqr = iLQRSolverOptions(verbose=false,
      iterations=500)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=0.001,
    cost_tolerance_intermediate=1.0e-4,
    iterations=30,
    penalty_scaling=2.0,
    penalty_initial=10.0)

num_lift = 3
prob_load = gen_prob(:load, quad_params, load_params)
prob_lift = [gen_prob(i, quad_params, load_params) for i = 1:3]
@time plift_al, pload_al, slift_al, sload_al = solve_admm_1slack(prob_lift,prob_load,:parallel,opts_al)


prob_load2 = gen_prob_all(quad_params, load_params, agent=:load)
prob_lift2 = [gen_prob_all(quad_params, load_params, agent=i) for i = 1:num_lift]
@time plift_al, pload_al, slift_al, sload_al = solve_admm_1slack(prob_lift2,prob_load2,:parallel,opts_al)

# include("visualization.jl")
# vis = Visualizer()
# open(vis)
visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al])
