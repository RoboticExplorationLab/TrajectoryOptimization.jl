using Random
using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
using BenchmarkTools

if nworkers() != 3
	addprocs(3,exeflags="--project=$(@__DIR__)")
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

num_lift = 3
quat = true
parallel = true
r0_load = [0, 0, 0.25]
prob_lift, prob_load = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al);

@time sol, sol_solver, sol_x = solve_admm(prob_lift, prob_load, quad_params,
    load_params, parallel, opts_al);


include("visualization.jl")
vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, sol,false)






using Plots
plot(n,tb)
plot!(n,tp,xlabel="number of lift agents",ylabel="time (s)")

using PGFPlots
const PGF = PGFPlots


tb = [1.355, 3.709, 5.165, 10.572, 11.565]
tp = [.717, .809, .895, 1.042, 1.148]

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
