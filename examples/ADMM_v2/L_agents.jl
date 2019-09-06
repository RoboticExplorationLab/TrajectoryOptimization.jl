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
using BenchmarkTools

include("problem.jl")
include("methods.jl")

# Solver options
verbose=true

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
prob_lift, prob_load = trim_conditions(num_lift,r0_load,quad_params,load_params,quat,opts_al)

# visualize_quadrotor_lift_system(vis, [[prob_load]; prob_lift])

@time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift, prob_load, quad_params,
    load_params, :parallel, opts_al, max_iters=10)


include("visualization.jl")
vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al],false)
