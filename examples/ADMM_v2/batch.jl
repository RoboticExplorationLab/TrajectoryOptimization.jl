using ForwardDiff, LinearAlgebra, Plots, StaticArrays
using Combinatorics
using Distributed
using TrajectoryOptimization
const TO = TrajectoryOptimization
include("visualization.jl")
include("problem.jl")
include("methods.jl")


# Solver options
verbose=false

opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=50)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=20,
    penalty_scaling=10.0,
    penalty_initial=1.0e-3)


# Create Problem
num_lift = 3
quat = true
r0_load = [-1.5,-1.5,0.25]
scenario = :slot

prob = gen_prob(:batch, quad_params, load_params, r0_load, scenario=scenario, num_lift=num_lift,quat=quat)
TO.has_quat(prob.model)
rollout!(prob)
prob.U
cost(prob)

# @btime solve($prob,$opts_al)
# prob = trim_conditions_batch(num_lift, r0_load, quad_params, load_params, quat, opts_al)
# @btime solve($prob, $opts_al)
solver = AugmentedLagrangianSolver(prob, opts_al)
solver.stats[:tstart] = time()
prob.obj[1]
@time solver = solve!(prob, solver)
solver.stats[:cost_uncon]
visualize_batch(vis,prob,scenario,num_lift)
solver.stats[:iterations]

vis = Visualizer()
open(vis)

max_violation(prob)
#=
Notes:
N lift is faster with trim conditions
Doorway is also faster with trim conditions
=#
