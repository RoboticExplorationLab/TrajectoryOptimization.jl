using Random
using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
if nworkers() != 3
	addprocs(3,exeflags="--project=$(@__DIR__)")
end

using TrajectoryOptimization
include("admm_solve.jl")
@everywhere using StaticArrays
@everywhere using TrajectoryOptimization
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))
@everywhere const TO = TrajectoryOptimization


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

function init_quad_ADMM(x0=[0, 0, 0.5], xf=[7.5, 0, 0.5]; distributed=true,quat=false)
		if distributed
			probs = ddata(T=Problem{Float64,Discrete});
			@sync for (j,w) in enumerate(workers())
				@spawnat w probs[:L] = build_quad_problem(j,x0,xf,quat)
			end
			prob_load = build_quad_problem(:load,x0,xf,quat)
		else
			probs = Problem{Float64,Discrete}[]
			prob_load = build_quad_problem(:load,x0,xf,quat)
			for i = 1:num_lift
				push!(probs, build_quad_problem(i,x0,xf,quat))
			end
		end
		return probs, prob_load
end
@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))
x0 = [1,   0.5,  0.5]
xf = [6.0, -0.5, 1.5]
probs, prob_load = init_quad_ADMM(x0, xf, distributed=true, quat=true);
@time sol,solvers = solve_admm(prob_load, probs, opts_al)

visualize_quadrotor_lift_system(vis, sol)

<<<<<<< HEAD
# vis = Visualizer()
=======
include("visualization.jl")
vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, sol)
>>>>>>> fe188aaa4a779dec7f4924727d509839523f32f2
if true
		TimerOutputs.reset_timer!()
		@time sol = solve_admm(prob_load, probs, opts_al)
		# visualize_quadrotor_lift_system(vis, sol)
		TimerOutputs.DEFAULT_TIMER
end
<<<<<<< HEAD
using MeshCat
vis = MeshCat.Visualizer()
open(vis)

r_cylinder = 0.5
_cyl = []
push!(_cyl,(3.75,1.,r_cylinder))
push!(_cyl,(3.75,-1.,r_cylinder))

visualize_quadrotor_lift_system(vis, sol, _cyl)
=======


function robustness_check(opts, nruns=50)
	Random.seed!(1)
	disable_logging(Logging.Info)

	x0 = [Uniform(-1,1), Uniform(-0.5,0.5), Uniform(0.5, 1.5)]
	xf = [Uniform(6,7.5), Uniform(-0.5,0.5), Uniform(0.5, 1.5)]
	stats = Dict{Symbol,Vector}(
		:c_max=>zeros(nruns),
		:iters=>zeros(Int,nruns),
		:time=>zeros(nruns),
		:x0=>[zeros(3) for i = 1:nruns],
		:xf=>[zeros(3) for i = 1:nruns],
		:success=>zeros(Bool,nruns),
	)
	for i = 1:nruns
		print("Sample $i:")
		x0_load = rand.(x0)
		xf_load = rand.(xf)
		# x0_load = [0, 0, 0.5]
		# xf_load = [7.5, 0, 0.5]
		probs, prob_load = init_quad_ADMM(x0_load, xf_load, distributed=true, quat=true)
		sol,solvers = solve_admm(prob_load, probs, opts)
		stats[:c_max][i] = solvers[1].stats[:viol_ADMM]
		stats[:iters][i] = solvers[1].stats[:iters_ADMM]
		stats[:x0][i] = x0_load
		stats[:xf][i] = xf_load
		stats[:success][i] = stats[:c_max][i] < opts.constraint_tolerance
		t = @elapsed solve_admm(prob_load, probs, opts)
		stats[:time][i] = t
		stats[:success][i] ? success = "success" : success = "failed"
		println(" $success ($t sec)")
	end
	return stats

end
robustness_check(opts_al, 10)


function export_traj(sol)
	for (i,prob) in enumerate(sol)
		if i == 1
			name = "load"
		else
			name = "quad$(i-1)"
		end
		open(name * ".txt", write=true) do f
			for k = 1:prob.N
				if i == 1
					inds = [1,2,3,4,5,6]
				else
					inds = [1,2,3,8,9,10]
				end
				for j in inds
					print(f, prob.X[k][j], " ")
				end
			end
		end
	end
end
export_traj(sol)
>>>>>>> fe188aaa4a779dec7f4924727d509839523f32f2
