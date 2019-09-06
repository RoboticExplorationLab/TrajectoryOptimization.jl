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
t_cache = Float64[]
t_cache_seq = Float64[]
num_lift_cache = collect(3:15)
# include("visualization.jl")
# vis = Visualizer()
# open(vis)

for num_lift = num_lift_cache
	@info "Initial solve $num_lift"
	# run first
	@time prob_lift, prob_load = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al);
	wait.([@spawnat w reset_control_reference!(prob_lift[:L]) for w in workers()[1:num_lift]])
	@time solve_admm(prob_lift, prob_load, quad_params,
	    load_params, parallel, opts_al);

	@info "Compiled solve  $num_lift"
	# run fast
	prob_lift, prob_load = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al);
	t = @belapsed begin
		wait.([@spawnat w reset_control_reference!(prob_lift[:L]) for w in workers()[1:num_lift]])
		solve_admm(prob_lift,prob_load, quad_params,load_params,parallel, opts_al);
	end

	if parallel
		push!(t_cache,t)
	else
		push!(t_cache_seq,t)
	end

	# visualize_quadrotor_lift_system(vis, sol,false)
end

t_cache_seq
t_cache
using Plots
plot(num_lift_cache,convert.(Float64,t_cache))
plot!(num_lift_cache,convert.(Float64,t_cache_seq))
plot(num_lift_cache[1:6],convert.(Float64,timing_batch),xlabel="number of lift agents",ylabel="time (s)")

timing_batch
num_lift_cache[1:6]
t_cache

###
using PGFPlots
const PGF = PGFPlots

num_lift_batch = [(3:8)...,(10:15)...] # 9 break for some reason...
tp = [.4996, .5095, .5153, .5371, .5374, .5601, .5507, .5653, .5623, .5625, .5816, .5750, .5765]
ts = [.5860, .5830, .6301, .5965, .6167, .6244, .5924, .6170, .6113, .6316, .6275, .6183, .6325]
tb = [1.521, 6.0266, 6.4757, 10.4257,10.5166, 13.476, 20.3489, 24.5767, 28.4216, 31.9788, 35.8373, 41.173]

_tp = PGF.Plots.Linear(num_lift_cache,tp,mark="",style="color=orange, thick",legendentry="parallel")
_ts = PGF.Plots.Linear(num_lift_cache,ts,mark="",style="color=green, thick",legendentry="sequential")
_tb = PGF.Plots.Linear(num_lift_batch,tb,mark="",style="color=blue, thick",legendentry="batch")

_tps = PGF.Plots.Scatter(num_lift_cache,tp, ["a" for kk in num_lift_cache], scatterClasses="{a={mark=*,orange,scale=1.0, mark options={fill=orange}}}")
_tss = PGF.Plots.Scatter(num_lift_cache,ts, ["a" for kk in num_lift_cache], scatterClasses="{a={mark=*,green,scale=1.0, mark options={fill=green}}}")
_tbs = PGF.Plots.Scatter(num_lift_batch,tb, ["a" for kk in num_lift_batch], scatterClasses="{a={mark=*,blue,scale=1.0, mark options={fill=blue}}}")

a = Axis([_tb;_ts;_tp;_tbs;_tss;_tps],
    xmin=3., ymin=0., xmax=15, ymax=45,
    axisEqualImage=false,
    hideAxis=false,
	style="grid=both",
	ylabel="time (s)",
	xlabel="number of lift agents",
	ymode="log",
	legendStyle="{at={(1.0,0.65)},anchor=north east}")

# Save to tikz format
dir = joinpath(pwd(),"examples/ADMM_v2/plots")
PGF.save(joinpath(dir,"L_lift.tikz"), a, include_preamble=false)

num_lift = 15
@time prob_lift, prob_load = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al);
wait.([@spawnat w reset_control_reference!(prob_lift[:L]) for w in workers()[1:num_lift]])
@time sol, sol_solver, xx = solve_admm(prob_lift, prob_load, quad_params,
	load_params, parallel, opts_al);


vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, sol, false)
settransform!(vis["/Cameras/default"], compose(Translation(3.5,-5, 5),LinearMap(RotX(0)*RotZ(-pi/2))))
