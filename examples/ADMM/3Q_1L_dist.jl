using Distributed
using DistributedArrays
using TimerOutputs
if nworkers() != 3
	addprocs(3,exeflags="--project=$(@__DIR__)")
end

using TrajectoryOptimization
include("admm_solve.jl")
@everywhere using TrajectoryOptimization
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere using StaticArrays
@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))
@everywhere const TO = TrajectoryOptimization


# Initialize problems
verbose = false
opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=500)
opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-5,
    iterations=10,
    penalty_scaling=2.0,
    penalty_initial=10.)

function init_quad_ADMM(;distributed=true,quat=false)
		if distributed
			probs = ddata(T=Problem{Float64,Discrete});
			@sync for (j,w) in enumerate(workers())
				@spawnat w probs[:L] = build_quad_problem(j,quat)
			end
			prob_load = build_quad_problem(:load,quat)
		else
			probs = Problem{Float64,Discrete}[]
			prob_load = build_quad_problem(:load,quat)
			for i = 1:num_lift
				push!(probs, build_quad_problem(i,quat))
			end
		end
		return probs, prob_load
end
probs, prob_load = init_quad_ADMM(distributed=false, quat=true);


@time sol = solve_admm(prob_load, probs, opts_al)

vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, sol)
if true
		TimerOutputs.reset_timer!()
		@time sol = solve_admm(prob_load, probs, opts_al)
		# visualize_quadrotor_lift_system(vis, [[prob_load]; probs], _cyl)
		TimerOutputs.DEFAULT_TIMER
end
