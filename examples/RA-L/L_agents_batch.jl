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
quat= true
r0_load = [0, 0, 0.25]

num_lift_batch = [(3:8)...,(10:15)...] # 9 break for some reason...
timing_batch = []

for nn = num_lift_batch
    #first solve
    @info "Initial solve $nn"
    prob = trim_conditions_batch(nn,r0_load,quad_params,load_params,quat,opts_al)
    @time solve(prob,opts_al)

    # fast solve
    @info "Compiled solve $nn"
    prob = trim_conditions_batch(nn,r0_load,quad_params,load_params,quat,opts_al)
    t = @belapsed begin
        solve($prob,$opts_al)
    end
    push!(timing_batch,t)
end

println(timing_batch)

# vis = Visualizer()
# open(vis)
# visualize_batch(vis,sol,obs,num_lift)

#=
Notes:
Fastest solve with midpoint cost = 10.0
Smoothest solution with midpoint cost = 1.0
=#
