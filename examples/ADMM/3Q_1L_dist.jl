using Random
using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
# using JSON3

if nworkers() != 3
	addprocs(3,exeflags="--project=$(@__DIR__)")
end
import TrajectoryOptimization: Discrete

using TrajectoryOptimization
const TO = TrajectoryOptimization
@everywhere using TrajectoryOptimization
@everywhere const TO = TrajectoryOptimization
include("admm_solve.jl")
@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_1_slack_problem.jl"))

function init_quad_ADMM(distributed=true)
	if distributed
		probs = ddata(T=Problem{Float64,Discrete});
		@sync for (j,w) in enumerate(workers())
			@spawnat w probs[:L] = gen_lift_problem(j)
		end
		prob_load = gen_lift_problem(:load)
	else
		probs = Problem{Float64,Discrete}[]
		prob_load = gen_lift_problem(:load)
		for i = 1:num_lift
			push!(probs, gen_lift_problem(i))
		end
	end
	return probs, prob_load
end

# Initialize problems
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

# @everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_1_slack_problem.jl"))
include("3Q_1L_1_slack_problem.jl")
parallel = false
probs, prob_load = init_quad_ADMM(parallel)#x0, xf, distributed=true, quat=true, infeasible=false, doors=false);
@time sol,solvers = solve_admm(prob_load, probs, opts_al, parallel=parallel)

include("visualization.jl")
vis = Visualizer()
open(vis)
anim = visualize_lift_system(vis, sol)
