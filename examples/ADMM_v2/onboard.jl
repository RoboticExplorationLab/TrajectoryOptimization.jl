using Random
# using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
using BenchmarkTools
import TrajectoryOptimization: Discrete

using TrajectoryOptimization
const TO = TrajectoryOptimization


function startup_workers(blas=true; localworkers=false)
    if localworkers
        addprocs(3,exeflags="--project=$(@__DIR__)")
    else    
        exename="/home/odroid/Julia/julia-1.0.4/bin/julia"
        addprocs([("odroid@rexquad3",1)], exename=exename, exeflags="--project=$(@__DIR__)",        enable_threaded_blas=blas)
        addprocs([("odroid@quad3",1)], exename=exename, exeflags="--project=$(@__DIR__)", enable_threaded_blas=blas)
        addprocs(1, exeflags="--project=$(@__DIR__)", enable_threaded_blas=blas)
    end 
end

if worker_includes
    @everywhere using TrajectoryOptimization
    @everywhere const TO = TrajectoryOptimization
    @everywhere using StaticArrays
    @everywhere using LinearAlgebra
    @everywhere using DistributedArrays
    @everywhere using ForwardDiff
    @everywhere include(joinpath(dirname(@__FILE__),"problem.jl"))
    @everywhere include(joinpath(dirname(@__FILE__),"methods_distributed.jl"))
    @everywhere include(joinpath(dirname(@__FILE__),"methods.jl"))
    @everywhere include(joinpath(dirname(@__FILE__),"models.jl"))
end
worker_includes = false

function init_dist(;quat=false)
	probs = ddata(T=Problem{Float64,Discrete});
	@sync for (j,w) in enumerate(workers())
		@spawnat w probs[:L] = gen_prob(j, quad_params, load_params, quat=quat)
	end
	prob_load = gen_prob(:load, quad_params, load_params, quat=quat)

	return probs, prob_load
end

function run_dist(quat)
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

    probs, prob_load = init_dist(quat=quat);
    @time sol, sol_solvers, xx = solve_admm(probs, prob_load, true, opts_al);
end

function run_serial(quat)
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
    prob_load = gen_prob(:load, quad_params, load_params, quat=quat)
    prob_lift = [gen_prob(i, quad_params, load_params, quat=quat) for i = 1:3]
    @time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift, prob_load, :sequential, opts_al)
end

function run_batch(quat)
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
    prob = gen_prob(:batch, quad_params, load_params, quat=quat)
    # prob = gen_prob_all(quad_params, load_params, agent=:batch)

    # @btime solve($prob,$opts_al)
    @time solve!(prob,opts_al)
end

