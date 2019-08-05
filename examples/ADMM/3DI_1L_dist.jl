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
@everywhere include(joinpath(dirname(@__FILE__),"3DI_problem.jl"))
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

function init_DI(distributed=true)
	if distributed
	    probs = ddata(T=Problem{Float64,Discrete});
	    @sync for (j,w) in enumerate(workers())
			@spawnat w probs[:L] = build_DI_problem(j)
	    end
	    prob_load = build_DI_problem(:load)
	else
	    probs = Problem{Float64,Discrete}[]
	    prob_load = build_DI_problem(:load)
	    for i = 1:num_lift
		push!(probs, build_DI_problem(i))
	    end
	end
	return probs, prob_load
end
@everywhere include(joinpath(dirname(@__FILE__),"3DI_problem.jl"))
probs, prob_load = init_DI(true);

@time sol = solve_admm(prob_load, probs, opts_al)
if false
	# @time sol = solve_admm(prob_load, probs, opts_al)

end
vis = Visualizer()
open(vis)
sol[1].model.info
visualize_DI_lift_system(vis, sol)
