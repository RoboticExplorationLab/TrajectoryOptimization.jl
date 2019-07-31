using Distributed
using TimerOutputs
addprocs(3)


using TrajectoryOptimization
include("admm_solve.jl")
@everywhere using TrajectoryOptimization
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include("examples/ADMM/3DI_problem.jl")
@everywhere const TO = TrajectoryOptimization




# Initialize problems
verbose = false
distributed = true
opts_ilqr = iLQRSolverOptions(verbose=verbose,iterations=500)
opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-5,
    iterations=100,
    penalty_scaling=2.0,
    penalty_initial=10.)
opts = opts_al

distributed = true
distributed = false
if distributed
    probs = ddata(T=Problem{Float64,Discrete});
    @sync for i in workers()
        j = i - 1
        @spawnat i probs[:L] = build_DI_problem(j)
    end
    prob_load = build_DI_problem(:load)
else
    probs = Problem{Float64,Discrete}[]
    prob_load = build_DI_problem(:load)
    for i = 1:num_lift
        push!(probs, build_DI_problem(i))
    end
end

@time sol = solve_admm(prob_load, probs, opts_al)

vis = Visualizer()
open(vis)
visualize_lift_system(vis, sol, r_lift, r_load)
