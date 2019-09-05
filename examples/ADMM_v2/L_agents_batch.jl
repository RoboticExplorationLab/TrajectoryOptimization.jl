using ForwardDiff, LinearAlgebra, Plots, StaticArrays
const TO = TrajectoryOptimization
include("visualization.jl")
include("problem.jl")
include("methods.jl")

# Solver options
verbose=true

opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=250)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=20,
    penalty_scaling=10.0,
    penalty_initial=1.0e-3)


# Create Problem
num_lift = 5
obs = false
quat= false
r0_load = [0, 0, 0.25]
prob = trim_conditions_batch(num_lift,r0_load,quad_params,load_params,quat,opts_al)

rollout!(prob)
prob.x0
# @btime solve($prob,$opts_al)
@time solve!(prob,opts_al)
# @btime solve($prob,$opts_al)

max_violation(prob)
TO.findmax_violation(prob)

vis = Visualizer()
open(vis)
visualize_batch(vis,prob,false,num_lift)

#=
Notes:
Fastest solve with midpoint cost = 10.0
Smoothest solution with midpoint cost = 1.0
=#
