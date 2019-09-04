using Random
using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs

if nworkers() != 3
	addprocs(3,exeflags="--project=$(@__DIR__)")
end
import TrajectoryOptimization: Discrete

using TrajectoryOptimization
const TO = TrajectoryOptimization
@everywhere using TrajectoryOptimization
@everywhere const TO = TrajectoryOptimization
@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include(joinpath(dirname(@__FILE__),"problem.jl"))
@everywhere include(joinpath(dirname(@__FILE__),"methods.jl"))


include("methods_distributed.jl")

verbose = false

opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=200)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-5,
    iterations=10,
    penalty_scaling=2.0,
    penalty_initial=10.)

probs, prob_load = init_dist();
@time sol, sol_solvers, xx = solve_admm_1slack_dist(probs, prob_load, true, opts_al);

vis = Visualizer()
open(vis)
include("visualization.jl")
visualize_quadrotor_lift_system(vis, sol)

idx = [(1:3)...,(7 .+ (1:3))...]
output_traj(sol[2],idx,joinpath(pwd(),"trajectoriestraj0.txt"))
output_traj(sol[3],idx,joinpath(pwd(),"trajectoriestraj1.txt"))
output_traj(sol[4],idx,joinpath(pwd(),"trajectoriestraj2.txt"))
println(sol[2].x0[1:3])
println(sol[3].x0[1:3])
println(sol[4].x0[1:3])
