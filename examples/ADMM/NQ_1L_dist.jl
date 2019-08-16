using Random
using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
using BenchmarkTools

num_lift = 3
nn = copy(num_lift)
if nworkers() != num_lift
	addprocs(num_lift,exeflags="--project=$(@__DIR__)")
end
import TrajectoryOptimization: Discrete

using TrajectoryOptimization
@everywhere using TrajectoryOptimization
@everywhere const TO = TrajectoryOptimization
include("admm_solve.jl")

@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))


function init_quad_ADMM(x0=[0, 0, 0.5], xf=[7.5, 0, 0.5]; quat=false, num_lift=3, obstacles=true, distributed=true, kwargs...)
	if distributed
		probs = ddata(T=Problem{Float64,Discrete});
		@sync for (j,w) in enumerate(workers())
			@spawnat w probs[:L] = build_quad_problem(j,x0,xf,quat,obstacles,num_lift; kwargs...)
		end
		prob_load = build_quad_problem(:load,x0,xf,quat,obstacles,num_lift; kwargs...)
	else
		probs = Problem{Float64,Discrete}[]
		prob_load = build_quad_problem(:load,x0,xf,quat,obstacles,num_lift; kwargs...)
		for i = 1:num_lift
			push!(probs, build_quad_problem(i,x0,xf,quat,obstacles,num_lift; kwargs...))
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

opts_altro = ALTROSolverOptions{Float64}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-4,
    resolve_feasible_problem=false,
    projected_newton=false,
    projected_newton_tolerance=1.0e-4)

@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))

# n-teamlift
x0 = [0., 0., 0.3]
xf = [6., 0., 0.3]

num_lift = nn
probs, prob_load = init_quad_ADMM(x0, xf, distributed=true, num_lift=num_lift, obstacles=false,quat=true, infeasible=false, doors=false);
sol,_solver = solve_admm(prob_load, probs, opts_al, true)
@btime sol,_solver = solve_admm(prob_load, probs, opts_al, true)

include("visualization.jl")
vis = Visualizer()
open(vis)
anim = visualize_quadrotor_lift_system(vis, sol, door=:false)
