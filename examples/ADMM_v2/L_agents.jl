using Random
# using Distributions
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

if nworkers() != 3
	addprocs(3,exeflags="--project=$(@__DIR__)")
end

# Solver options
verbose=true

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

num_lift = 4
quat = true
r0_load = [0, 0, 0.25]
# prob_lift, prob_load = trim_conditions(num_lift,r0_load,quad_params,load_params,quat,opts_al)
a, b = trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts_al)
# prob_load = gen_prob(:load, quad_params, load_params, r0_load,
#     num_lift=num_lift, quat=quat, scenario=scenario)
# prob_lift = [gen_prob(i, quad_params, load_params, r0_load,
#     num_lift=num_lift, quat=quat, scenario=scenario) for i = 1:num_lift]
# visualize_quadrotor_lift_system(vis, [[prob_load]; prob_lift])

@time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift, prob_load, quad_params,
    load_params, :parallel, opts_al, max_iters=10)


include("visualization.jl")
vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, [[pload_al]; plift_al],false)


function init_dist2(;quat=false)
	probs = ddata(T=Problem{Float64,Discrete});
	@sync for (j,w) in enumerate(workers())
		@spawnat w probs[:L] = gen_prob(j, quad_params, load_params, quat=quat)
	end
	# prob_load = gen_prob(:load, quad_params, load_params, quat=quat)

	return probs, prob_load
end

init_dist2()

function init_dist(;quat=false)
	probs = ddata(T=Problem{Float64,Discrete});
	@sync for (j,w) in enumerate(workers())
		@spawnat w probs[:L] = gen_prob(j, quad_params, load_params, quat=quat)
	end
	prob_load = gen_prob(:load, quad_params, load_params, quat=quat)

	return probs, prob_load
end

function trim_conditions_dist(num_lift,r0_load,quad_params,load_params,quat,opts)
    scenario = :hover
    prob_load_trim = gen_prob(:load, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario)
    prob_lift_trim = [gen_prob(i, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario) for i = 1:num_lift]

    lift_trim, load_trim, slift_al, sload_al = solve_admm(prob_lift_trim, prob_load_trim, quad_params,
            load_params, :sequential, opts_al, max_iters=5)

    scenario = :p2p
    prob_load = gen_prob(:load, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario)
    initial_controls!(prob_load,[load_trim.U[end] for k = 1:prob_load.N])



    prob_lift = [gen_prob(i, quad_params, load_params, r0_load,
        num_lift=num_lift, quat=quat, scenario=scenario) for i = 1:num_lift]

    for i = 1:num_lift
        prob_lift[i].x0[4:7] = lift_trim[i].X[end][4:7]
        prob_lift[i].xf[4:7] = lift_trim[i].X[end][4:7]
        initial_controls!(prob_lift[i],[lift_trim[i].U[end] for k = 1:prob_lift[i].N])
        # rollout!(prob_lift[i])
    end

	# prob_lift_copy = [@spawnat w copy(prob_lift) for w in workers()]
	probs = ddata(T=Problem{Float64,Discrete});
	@sync for (j,w) in enumerate(workers())
		@spawnat w probs[:L] = copy(prob_lift)
	end

    return probs, prob_load
end














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
