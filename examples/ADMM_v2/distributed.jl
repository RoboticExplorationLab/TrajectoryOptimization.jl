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
include("methods.jl")
@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include(joinpath(dirname(@__FILE__),"problem.jl"))

function init_dist()
	probs = ddata(T=Problem{Float64,Discrete});
	@sync for (j,w) in enumerate(workers())
		@spawnat w probs[:L] = gen_prob(j)
	end
	prob_load = gen_prob(:load)

	return probs, prob_load
end

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

probs, prob_load = init_dist()
