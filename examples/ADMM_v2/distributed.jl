using Random
# using Distributions
using Logging
using Distributed
using DistributedArrays
using TimerOutputs
using BenchmarkTools
using TrajectoryOptimization
using Interpolations
using Plots

if nworkers() != 3
	addprocs(3,exeflags="--project=$(@__DIR__)")
end
addworkers(n) = addprocs(n,exeflags="--project=$(@__DIR__)")
import TrajectoryOptimization: Discrete

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

function init_dist(;num_lift=3, quat=false, scenario=:doorway, r0_load=[0,0,0.25])
	probs = ddata(T=Problem{Float64,Discrete});
	@show length(probs)
	@sync for (j,w) in enumerate(worker_quads(num_lift))
		@spawnat w probs[:L] = gen_prob(j, quad_params, load_params, r0_load,
			num_lift=num_lift, quat=quat, scenario=scenario)
	end
	prob_load = gen_prob(:load, quad_params, load_params, r0_load,
		num_lift=num_lift, quat=quat, scenario=scenario)

	return probs, prob_load
end

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
    penalty_initial=10.,
	cache_trajectories=true)

quat = true
num_lift = 3
scenario = :slot
scenario == :doorway ? obs = true : obs = false
r0_load = [-0.5, 1.5, 0.25]
probs, prob_load = init_dist(num_lift=num_lift, quat=quat, scenario=scenario, r0_load=r0_load);
wait.([@spawnat w reset_control_reference!(probs[:L]) for w in worker_quads(num_lift)])
@time sol, sol_solvers, solvers_init, xx = solve_admm(probs, prob_load, quad_params,
	load_params, true, opts_al, max_iters=2);
visualize_quadrotor_lift_system(vis, sol, scenario)




all(diff(times) .> 0)

sum(Js)
maximum(c_maxes)



function create_batch_trajectories(sol, sol_solvers, iter)
	num_lift = length(sol_solvers[2:end])
	n_load, m_load, N = size(sol[1])
	n_lift, m_lift = size(sol[2])
	n_batch = n_lift*num_lift + n_load
	m_batch = m_lift*num_lift + m_load
	X = [zeros(n_batch) for k = 1:N]
	U = [zeros(m_batch) for k = 1:N-1]

	x_traj = [solver.stats[:X][iter] for solver in sol_solvers]
	u_traj = [solver.stats[:U][iter] for solver in sol_solvers]
	for k = 1:N-1
		x_load = x_traj[1][k]
		u_load = u_traj[1][k]
		x_lift = [x[k] for x in x_traj[2:end]]
		u_lift = [u[k] for u in u_traj[2:end]]
		X[k] .= [vcat(x_lift...); x_load]
		U[k] .= [vcat(u_lift...); u_load]
	end
	x_load = x_traj[1][N]
	x_lift = [x[N] for x in x_traj[2:end]]
	X[N] .= [vcat(x_lift...); x_load]

	return X, U
end
X, U = create_batch_trajectories(sol, sol_solvers, 1)

cost(prob.obj, X, U, TO.get_dt_traj(prob)) ≈ sum(Js)
max_violation(prob, X, U)


function interp_trajectories(solvers)
	num_agents = length(solvers)

	times = [copy(sol.stats[:timestamp]) for sol in solvers]
	final_time = maximum([t[end] for t in times[2:end]])
	[push!(t, final_time) for t in times[2:end]]
	max_iter = maximum(length.(times[2:end]))
	for i = 2:num_agents
		fillend!(times[i], max_iter)
	end
	t_sample = unique_approx(vcat(times[2:end]...))
	for t in t_sample


end
t_sample = interp_trajectories(sol_solvers)

function plot_convergence(solvers, solvers_init)
	num_agents = length(solvers)
	Js = Float64[]
	time = Float64[]
	c_max = Float64[]

	# Initial cost
	J0 = sum([sol.stats[:cost_uncon][1] for sol in solvers_init])
	push!(Js, J0)
	push!(time, 0.0)
	push!(c_max, NaN)

	# Cost after initial solve
	J = sum([sol.stats[:cost_uncon][end] for sol in solvers_init])
	t = maximum([sol.stats[:timestamp][end] for sol in solvers_init])
	push!(Js, J)
	push!(time, t)
	push!(c_max, NaN)

	# Parallel solve
	costs = [copy(sol.stats[:cost_uncon]) for sol in solvers]
	c_maxes = [copy(sol.stats[:c_max]) for sol in solvers]
	times = [copy(sol.stats[:timestamp]) for sol in solvers]
	final_time = maximum([t[end] for t in times[2:end]])
	[push!(t, final_time) for t in times[2:end]]
	max_iter = maximum(length.(times[2:end]))
	for i = 2:num_agents
		fillend!(costs[i], max_iter)
		fillend!(c_maxes[i], max_iter)
		fillend!(times[i], max_iter)
	end
	cost_interps = [LinearInterpolation(times[i], costs[i]) for i = 2:num_agents]
	# return cost_interps
	t_sample = unique_approx(vcat(times[2:end]...))
	costs_interp = [LinearInterpolation(times[i],   costs[i])(t_sample) for i = 2:num_agents]
	c_max_interp = [LinearInterpolation(times[i], c_maxes[i])(t_sample) for i = 2:num_agents]
	total_cost = map(sum,     zip(costs_interp...))
	total_cmax = map(maximum, zip(c_max_interp...))

	append!(Js,    total_cost)
	append!(c_max, total_cmax)
	append!(time,  t_sample)

	# Load solve
	@assert times[1][1] > time[end]
	append!(Js, costs[1])
	append!(c_max, c_maxes[1])
	append!(time,  times[1])


	# return total_cost, total_cmax, t_sample
	# append!(Js, costs)
	# append!(c_max, c_maxes)
	# append!(time, times)
	return Js, c_max, time
end
costs, c_maxes, times = plot_convergence(sol_solvers, solvers_init)
plot(times, costs, yscale=:log10)
sol[2].obj
t_sample = unique_approx(vcat(times[2:end]...))
LinearInterpolation(times[2], costs[2])(t_sample)
costs

function fillend!(a,N)
	n = length(a)
	if n < N
		tail = ones(N-n)*a[end]
		append!(a, tail)
	end
end

function unique_approx(a, tol=0.01)
	adiff = [sort!(a); Inf]
	return a[diff(adiff) .> tol]
end



@btime begin
	wait.([@spawnat w reset_control_reference!(probs[:L]) for w in workers()])
	@time solve_admm($probs, $prob_load, $quad_params, $load_params, true, $opts_al);
end

@btime solve_admm($probs, $prob_load, $quad_params, $load_params, $true, $opts_al);

X_cache, U_cache, X_lift, U_lift = init_cache(prob_load, probs);
fetch(@spawnat 1 length(X_cache[:L]))
cache = (X_cache=X_cache, U_cache=U_cache, X_lift=X_lift, U_lift=U_lift);


include("visualization.jl")
vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, sol, obs)
settransform!(vis["/Cameras/default"], compose(Translation(-7,0, -3),LinearMap(RotX(0)*RotZ(0))))

# grab frames
settransform!(vis["/Cameras/default"], compose(Translation(3.5, 0, 6.),LinearMap(RotX(0)*RotY(-pi/3))))

kk = [1,13,26,39,51]
k = 51
n_slack = 3
x_load = sol[1].X[k][1:n_slack]

for i = 1:num_lift
	x_lift = sol[i+1].X[k][1:n_slack]
	settransform!(vis["cable"]["$i"], cable_transform(x_lift,x_load))
	settransform!(vis["lift$i"], compose(Translation(x_lift...),LinearMap(Quat(sol[i+1].X[k][4:7]...))))
end
settransform!(vis["load"], Translation(x_load...))
##

idx = [(1:3)...,(7 .+ (1:3))...]
output_traj(sol[2],idx,joinpath(pwd(),"examples/ADMM_v2/trajectories/traj0.txt"))
output_traj(sol[3],idx,joinpath(pwd(),"examples/ADMM_v2/trajectories/traj1.txt"))
output_traj(sol[4],idx,joinpath(pwd(),"examples/ADMM_v2/trajectories/traj2.txt"))
println(sol[2].x0[1:3])
println(sol[3].x0[1:3])
println(sol[4].x0[1:3])
end

pwd()
output_traj(sol[2],idx,joinpath(pwd(),"examples/ADMM_v2/trajectories/traj0.txt"))
output_traj(sol[3],idx,joinpath(pwd(),"examples/ADMM_v2/trajectories/traj1.txt"))
output_traj(sol[4],idx,joinpath(pwd(),"examples/ADMM_v2/trajectories/traj2.txt"))
println(sol[2].x0[1:3])
println(sol[3].x0[1:3])
println(sol[4].x0[1:3])

include("visualization.jl")
vis = Visualizer()
open(vis)
visualize_quadrotor_lift_system(vis, sol)
