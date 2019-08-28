using Random
using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
using BenchmarkTools

num_lift = 3
if nworkers() != num_lift
	addprocs(num_lift,exeflags="--project=$(@__DIR__)")
end
using TrajectoryOptimization
import TrajectoryOptimization: Discrete

@everywhere using TrajectoryOptimization
@everywhere const TO = TrajectoryOptimization
include("admm_solve.jl")

@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))


function init_lift_ADMM(lift_model::Model, x0=[0, 0, 0.5], xf=[7.5, 0, 0.5]; quat=false, num_lift=3, obstacles=true, distributed=true, kwargs...)
	if distributed
		probs = ddata(T=Problem{Float64,Discrete});
		@sync for (j,w) in enumerate(workers())
			@spawnat w probs[:L] = build_lift_problem(lift_model, j,x0,xf,quat,obstacles,num_lift; kwargs...)
		end
		prob_load = build_lift_problem(lift_model, :load,x0,xf,quat,obstacles,num_lift; kwargs...)
	else
		probs = Problem{Float64,Discrete}[]
		prob_load = build_lift_problem(lift_model, :load, x0,xf,quat,obstacles,num_lift; kwargs...)
		for i = 1:num_lift
			push!(probs, build_lift_problem(lift_model, i, x0,xf,quat,obstacles,num_lift; kwargs...))
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

@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))

# n-teamlift
x0 = [0., 0., 0.3]
xf = [6., 0., 0.3]

lift_model = quadrotor_lift
lift_model = doubleintegrator3D_lift
probs, prob_load = init_lift_ADMM(lift_model, x0, xf,
	distributed=false, num_lift=num_lift, obstacles=false,quat=true, infeasible=false, doors=false);
sol0, solvers0 = solve_admm(prob_load, probs, opts_al, parallel=true, max_iter=2)

include("visualization.jl")
vis = Visualizer()
open(vis)
anim = visualize_lift_system(vis, sol0, door=:false)

# timing results
n = [2, 3, 4, 5, 6]
tb = [1.355, 3.709, 5.165, 10.572, 11.565]
tp = [.717, .809, .895, 1.042, 1.148]

using Plots
plot(n,tb)
plot!(n,tp,xlabel="number of lift agents",ylabel="time (s)")

using PGFPlots
const PGF = PGFPlots

_tb = PGF.Plots.Linear(n,tb,mark="o",style="color=orange, thick")
_tp = PGF.Plots.Linear(n,tp,mark="o",style="color=blue, thick")

_tbs = PGF.Plots.Scatter(n,tb,style="color=orange, thick")
_tps = PGF.Plots.Scatter(n,tp,style="color=blue, thick")

_tbs = PGF.Plots.Scatter(n,tb, ["a" for kk in n], scatterClasses="{a={mark=*,orange,scale=1.0, mark options={fill=orange}}}")
_tps = PGF.Plots.Scatter(n,tp, ["a" for kk in n], scatterClasses="{a={mark=*,blue,scale=1.0, mark options={fill=blue}}}")

a = Axis([_tb;_tp;_tbs;_tps],
    xmin=2., ymin=0., xmax=6, ymax=12.,
    axisEqualImage=false,
    hideAxis=false,
	style="grid=both",
	ylabel="time (s)",
	xlabel="number of lift agents")

# Save to tikz format
paper = "/home/taylor/Research/distributed_team_lift_paper/images"
PGF.save(joinpath(paper,"n_lift.tikz"), a, include_preamble=false)
