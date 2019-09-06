using ForwardDiff, LinearAlgebra, Plots, StaticArrays
const TO = TrajectoryOptimization
using BenchmarkTools

include("visualization.jl")
include("problem.jl")
include("methods.jl")

# Solver options
verbose=false

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
obs = false
quat= false
r0_load = [0, 0, 0.25]

num_lift = collect(3:4)
timing = []
for nn = 3:15
    prob = trim_conditions_batch(nn,r0_load,quad_params,load_params,quat,opts_al)
    solve(prob,opts_al)
    t = @belapsed solve($prob,$opts_al)
    push!(timing,t)
end

# vis = Visualizer()
# open(vis)
# visualize_batch(vis,sol,obs,num_lift)

#=
Notes:
Fastest solve with midpoint cost = 10.0
Smoothest solution with midpoint cost = 1.0
=#
