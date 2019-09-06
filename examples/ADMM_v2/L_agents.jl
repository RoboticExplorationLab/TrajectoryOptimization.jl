using Random
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
using BenchmarkTools

if nworkers() != 15
	addprocs(15,exeflags="--project=$(@__DIR__)")
end
import TrajectoryOptimization: Discrete

using TrajectoryOptimization
const TO = TrajectoryOptimization
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

# Solver options
verbose=false

opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=500)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=0.001,
    cost_tolerance_intermediate=1.0e-4,
    iterations=30,
    penalty_scaling=2.0,
    penalty_initial=10.)

quat = true
parallel = true
r0_load = [0, 0, 0.25]
t_cache = []
# num_lift_cache = collect(3:4)
# include("visualization.jl")
# vis = Visualizer()
# open(vis)
num_lift = 4

prob_lift, prob_load = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al);
wait.([@spawnat w reset_control_reference!(prob_lift[:L]) for w in workers()[1:num_lift]])
@time solve_admm(prob_lift, prob_load, quad_params,
    load_params, parallel, opts_al);

copy_probs(prob_lift);

for num_lift = 3:4

	# prob_lift, prob_load = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al);
	# wait.([@spawnat w reset_control_reference!(prob_lift[:L]) for w in workers()[1:num_lift]])
	# @time solve_admm(prob_lift, prob_load, quad_params,
	#     load_params, parallel, opts_al);

	prob_lift, prob_load = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al);
	t = @elapsed begin
		wait.([@spawnat w reset_control_reference!(prob_lift[:L]) for w in workers()[1:num_lift]])
		solve_admm(prob_lift, prob_load, quad_params,
	    load_params, parallel, opts_al);
	end
	push!(t_cache,t)

	# visualize_quadrotor_lift_system(vis, sol,false)
end

t_cache
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
