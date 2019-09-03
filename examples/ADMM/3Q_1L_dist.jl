using Random
using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
using JSON3
using Plots
using PGFPlots
const PGF = PGFPlots
if nworkers() != 3
	addprocs(3,exeflags="--project=$(@__DIR__)")
end
import TrajectoryOptimization: Discrete
const TO = TrajectoryOptimization

using TrajectoryOptimization
@everywhere using TrajectoryOptimization
@everywhere const TO = TrajectoryOptimization
include("admm_solve.jl")
@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using DistributedArrays
@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))

include("3Q_1L_problem.jl")
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

function change_door!(xf, door)
	door_width = 1.0
	if door == :middle
		xf[2] = 0.0
	elseif door == :left
		xf[2] = door_width
	elseif door == :right
		xf[2] = -door_width
	end
end

# Initialize problems
verbose = true
opts_ilqr = iLQRSolverOptions(verbose=true,
      iterations=500)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=0.001,
    cost_tolerance_intermediate=1.0e-4,
    iterations=30,
    penalty_scaling=2.0,
    penalty_initial=10.0)

@everywhere include(joinpath(dirname(@__FILE__),"3Q_1L_problem.jl"))
x0 = [0, 0,  4/6]
xf = [10., 0., 4/6]  # height of table: 0.75 (+0.2) m, spread height: 1.7
door = :middle
# change_door!(xf, door)
probs, prob_load = init_quad_ADMM(x0, xf, obstacles=false,distributed=false, quat=false, infeasible=false, doors=false);

@time sol,solvers = solve_admm(prob_load, probs, opts_al, parallel=false)

include("visualization.jl")
vis = Visualizer()
open(vis)
anim = visualize_lift_system(vis, sol, door=door)
# sol
# findmax_violation.(sol)
# @which max_violation(sol[2])
# @which max_violation(solvers[4])
#
#
# # Change the door partway through
# k_init = 30  # time step to change the door
# door2 = :left
# change_door!(xf, door2)
# probs, prob_load = init_quad_ADMM(sol[1].X[k_init][1:3], xf, distributed=false, quat=true, infeasible=false, doors=true);
# @time sol2,solvers2 = solve_admm(prob_load, probs, opts_al)
# anim = visualize_quadrotor_lift_system(vis, sol2, door=door2)
# plot_quad_scene(vis, 33, sol)
#
# visualize_door_change(vis, sol, sol2, door, door2, k_init)
#
#
# MeshCat.convert_frames_to_video("/home/bjack205/Downloads/meshcat_doorswitch_pedestal.tar", "quad_doorswitch.mp4"; overwrite=true)
#
#
# # n-teamlift
# x0 = [0., 0.,  0.3]
# xf = [6., 0., 0.3]
# probs, prob_load = init_quad_ADMM(x0, xf, distributed=true, num_lift=4, obstacles=false,quat=true, infeasible=false, doors=false);
# @time sol,solvers = solve_admm(prob_load, probs, opts_al, true)
# anim = visualize_quadrotor_lift_system(vis, sol, door=:false)
#
#
# include("visualization.jl")
# vis = Visualizer()
# open(vis)
# visualize_quadrotor_lift_system(vis, sol)
# # ghost_quadrotor_lift_system(vis, sol, [1,50,100], [1.0,0.5,1.0]; door=:middle, n_slack=3)
#
# # generate frame plots
# settransform!(vis["/Cameras/default"], compose(Translation(5., 0., 5.),LinearMap(RotY(-pi/3))))
# k = [1,25+1,50+1,75+1,sol[1].N]
# plot_quad_scene(vis, k[5], sol)
#
# # Z height plot
# p = plot(xlabel="time (s)",ylabel="height")
# N = sol[1].N
# _labels=["agent 1","agent 2", "agent 3"]
#
# tspan = range(0,stop=sol[1].tf,length=sol[1].N)
# for i = 1:num_lift
# 	z = [sol[i+1].X[k][3] for k = 1:N]
# 	p = plot!(tspan,z,label=_labels[i],color=i,width=2)
# end
# z = [sol[1].X[k][3] for k = 1:N]
# p = plot!(tspan,z,label="load",legend=:topleft,color=num_lift+1)
#
# for kk in k
# 	for i = 1:num_lift
# 		z = [sol[i+1].X[kk][3] for k = 1:N]
# 		p = plot!((tspan[kk],z[kk]),marker=:circle,color=i,label="")
# 		println(z[end])
# 	end
# 	z = [sol[1].X[kk][3] for k = 1:N]
# 	p = plot!((tspan[kk],z[kk]),label="",marker=:circle,color=num_lift+1)
# end
# display(p)
#
# using PGFPlots
# const PGF = PGFPlots
#
# _colors = ["blue","orange","green","purple"]
# _labels = ["load","agent 1","agent 2", "agent 3"]
# # Plot the trajectories
# z_plot = [PGF.Plots.Linear(tspan,[sol[i].X[k][3] for k = 1:N],
#     legendentry="$(_labels[i])",
#     mark="none",
#     style="color=$(_colors[i]), thick") for i = 1:length(sol)]
#
# tspan_points = [tspan[kk] for kk in k]
# zpoints = [[sol[i].X[kk][3] for kk in k] for i in 1:length(sol)]
# zz = ["a"]
# sc = "{l={mark=o,red, scale=1.5, mark options={fill=red},legendentry=""},
# 		a={mark=o,red,scale=1.5, mark options={fill=red}},
# 		b={mark=o,red,scale=1.5, mark options={fill=red}},
# 		c={mark=o,red,scale=1.5, mark options={fill=red}}}"
#
# z_markers = [PGF.Plots.Scatter(tspan_points, zpoints[i], ["a" for kk in k], scatterClasses="{a={mark=*,$(_colors[i]),scale=1.0, mark options={fill=$(_colors[i])}}}") for i = 1:length(sol)]
#
# # Plot the whole thing
# a = Axis([z_plot[2]; z_plot[3]; z_plot[4]; z_plot[1];z_markers[2];z_markers[3];z_markers[4];z_markers[1]],
#     xmin=0., ymin=0., xmax=sol[1].tf, ymax=2.1,
#     axisEqualImage=false,
#     legendPos="north west",
#     hideAxis=false,
# 	style="grid=both",
# 	xlabel="time (s)",
# 	ylabel="height (m)")
#
# # Save to tikz format
# paper = "/home/taylor/Research/distributed_team_lift_paper/images"
# PGF.save(joinpath(paper,"height.tikz"), a, include_preamble=false)
#
#
# if true
# 	TimerOutputs.reset_timer!()
# 	@time sol = solve_admm(prob_load, probs, opts_al)
# 	# visualize_quadrotor_lift_system(vis, sol)
# 	TimerOutputs.DEFAULT_TIMER
# end
#
#
# function robustness_check(opts, filename=joinpath(dirname(@__FILE__), "robustness_check.txt"))
# 	Random.seed!(1)
# 	disable_logging(Logging.Info)
#
# 	x0_bnd = [(-1, 0.75), (-4, 4), (0.3, 0.3)]
# 	dx = 0.2
# 	x0 = [range(x0_bnd[1]..., step=dx),
# 	      range(x0_bnd[2]..., step=dx),
# 	      range(x0_bnd[3]..., step=dx)]
#     sizes = length.(x0)
# 	n_points = prod(sizes)
# 	println("Total points: $n_points ($(join(sizes,'x')))")
# 	xf = [6.0, 0.0, .95]
#
# 	nruns = n_points
# 	stats = Dict{Symbol,Vector}(
# 		:c_max=>zeros(nruns),
# 		:iters=>zeros(Int,nruns),
# 		:time=>zeros(nruns),
# 		:x0=>[zeros(3) for i = 1:nruns],
# 		:xf=>[zeros(3) for i = 1:nruns],
# 		:success=>zeros(Bool,nruns),
# 		:gridsize=>sizes
# 	)
# 	i= 1
# 	for x in x0[1], y in x0[2], z in x0[3]
# 		print("Sample $i / $n_points ($x, $y, $z):")
# 		x0_load = [x,y,z]
# 		xf_load = xf
# 		# x0_load = [0, 0, 0.5]
# 		# xf_load = [7.5, 0, 0.5]
# 		probs, prob_load = init_quad_ADMM(x0_load, xf_load, distributed=true, quat=true)
# 		sol,solvers = solve_admm(prob_load, probs, opts)
# 		stats[:c_max][i] = solvers[1].stats[:viol_ADMM]
# 		stats[:iters][i] = solvers[1].stats[:iters_ADMM]
# 		stats[:x0][i] = x0_load
# 		stats[:xf][i] = xf_load
# 		stats[:success][i] = stats[:c_max][i] < opts.constraint_tolerance
# 		t = @elapsed solve_admm(prob_load, probs, opts)
# 		stats[:time][i] = t
# 		stats[:success][i] ? success = "success" : success = "failed"
# 		println(" $success ($t sec)")
# 		i+= 1
# 	end
#
# 	# # Write results
# 	# jsontxt = JSON3.write(stats)
# 	# open(filename, write=true) do f
# 	# 	write(f, jsontxt)
# 	# end
#
# 	return stats
# end
# stats = robustness_check(opts_al)
#
# # # Write results
# filename = joinpath(dirname(@__FILE__), "robustness_check.txt")
# jsontxt = JSON3.write(stats)
# open(filename, write=true) do f
# 	write(f, jsontxt)
# end
#
# # Read results back in
# stats2 = open(joinpath(dirname(@__FILE__), "robustness_check.txt"), read=true) do f
# 	JSON3.read(read(f))
# end
#
# # Create PGF Plot
# pushPGFPlotsOptions("scale=2.0")
# xs = [x[1] for x in stats[:x0]]
# ys = [x[2] for x in stats[:x0]]
# nx = length(unique(xs))
# ny = length(unique(ys))
#
# s = stats[:success]
# tmin = minimum(stats[:time])
# pts_succ = PGF.Plots.Scatter(ys[s], -xs[s], stats[:time][s], style="mark=*", legendentry="success")
# pts_fail = PGF.Plots.Scatter(ys[.!s], -xs[.!s], stats[:time][.!s], style="mark=diamond*", legendentry="failure")
# color_wall = "red"
# style = "color=$color_wall, fill=$color_wall"
# obs = [PGF.Plots.Circle(circle[2], -circle[1], circle[3], style=style) for circle in quad_obstacles()[1]]
# goal = let circle = stats[:xf][1]
# 	PGF.Plots.Circle(circle[2], -circle[1], 0.1, style="color=green, fill=green")
# end
# a = Axis([pts_succ; pts_fail; goal; obs],
# 	xmin=-3.5, xmax=3.5,
# 	ymin=-6.2, ymax=1,
# 	axisEqualImage=true,
# 	hideAxis=true,
# 	colorbar=true,
# 	legendPos="south west",
# 	style="{point meta max=10,
# 			colorbar style={
# 				ylabel=runtime (s),
# 				ytick={$tmin,2,4,6,8}, extra y ticks={10},
# 				 extra y tick labels={10+}
# 			 }}")
# PGF.save("initial_conditions.tikz", a, include_preamble=false)
#
# save(joinpath(paper,"escape_c_max.tikz"), a, include_preamble=false)
#
# ColorMaps.Named()
# x = [1,2,3]
# y = [2,4,1]
# plot(x, y)
