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
using PGFPlots
using PGFPlots
const PGF = PGFPlots

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

# Scenario
quat = true
num_lift = 3
scenario = :doorway
scenario == :doorway ? obs = true : obs = false
r0_load = [0, 1.5, 0.25]

# Solve distributed
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
    penalty_initial=10.,
	cache_trajectories=true)

probs, prob_load = init_dist(num_lift=num_lift, quat=quat, scenario=scenario, r0_load=r0_load);
wait.([@spawnat w reset_control_reference!(probs[:L]) for w in worker_quads(num_lift)])
@time sol, sol_solvers, solvers_init, xx = solve_admm(probs, prob_load, quad_params,
	load_params, true, opts_al, max_iters=1);
@btime begin
	wait.([@spawnat w reset_control_reference!($probs[:L]) for w in worker_quads($num_lift)])
	@time solve_admm($probs, $prob_load, $quad_params,
		$load_params, true, $opts_al, max_iters=1);
end

visualize_quadrotor_lift_system(vis, sol, scenario)

# Solve batch problem
opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=50)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=20,
    penalty_scaling=10.0,
    penalty_initial=1.0e-3,
	cache_trajectories=true)

prob = gen_prob(:batch, quad_params, load_params, r0_load, scenario=scenario, num_lift=num_lift, quat=quat)

@time begin
	solver = AugmentedLagrangianSolver(prob, opts_al)
	solver.stats[:tstart] = time()
	sol_b, solver = solve(prob, solver)
end
@btime begin
	solver = AugmentedLagrangianSolver($prob, $opts_al)
	$solver.stats[:tstart] = time()
	solve($prob, $solver)
end


visualize_batch(vis,sol_b,scenario,num_lift)
slot_obstacles()



# Generate convergence plot
function sample_traj(t, ts, Xs)
	ind = findfirst(x-> x >= t, ts)
	if isnothing(ind)
		return Xs[end]
	elseif ind > 1
		return Xs[ind-1]
	else
		return Xs[1]
	end
end

function fillend!(a,N)
	n = length(a)
	if n < N
		tail = ones(N-n)*a[end]
		append!(a, tail)
	end
end

function create_batch_trajectories(X_load, U_load, X_lift, U_lift)
	num_lift = length(X_lift)
	N = length(X_load)
	n_lift = length(X_lift[1][1])
	m_lift = length(U_lift[1][1])
	n_load = length(X_load[1])
	m_load = length(U_load[1])
	n_batch = n_lift*num_lift + n_load
	m_batch = m_lift*num_lift + m_load
	X_batch = [zeros(n_batch) for k = 1:N]
	U_batch = [zeros(m_batch) for k = 1:N-1]

	for k = 1:N-1
		x_lift = [x[k] for x in X_lift]
		u_lift = [u[k] for u in U_lift]
		X_batch[k] .= [vcat(x_lift...); X_load[k]]
		U_batch[k] .= [vcat(u_lift...); U_load[k]]
	end
	x_lift = [x[N] for x in X_lift]
	X_batch[N] .= [vcat(x_lift...); X_load[N]]
	return X_batch, U_batch
end

function recalc_stats(prob, solvers, solvers_init)
	num_agents = length(solvers)
	n_batch, m_batch, N = size(prob)
	dt_traj = TO.get_dt_traj(prob)

	Js = Float64[]
	time = Float64[]
	c_maxes = Float64[]
	labels = Symbol[]

	X_batch = [zeros(n_batch) for k = 1:N]
	U_batch = [zeros(m_batch) for k = 1:N-1]

	# Initial cost
	X_load = solvers_init[1].stats[:X0]
	U_load = solvers_init[1].stats[:U0]
	X_lift = [solver.stats[:X0] for solver in solvers_init[2:end]]
	U_lift = [solver.stats[:U0] for solver in solvers_init[2:end]]
	X_batch, U_batch = create_batch_trajectories(X_load, U_load, X_lift, U_lift)
	initial_controls!(prob, U_batch)
	rollout!(prob)
	J0 = cost(prob)
	c_max = max_violation(prob)
	push!(Js, J0)
	push!(c_maxes, c_max)
	push!(time, 0.0)
	push!(labels, :presolve)

	# Cost after initial solve
	X_lift = [solver.stats[:X][end] for solver in solvers_init[2:end]]
	U_lift = [solver.stats[:U][end] for solver in solvers_init[2:end]]
	X_load = solvers_init[1].stats[:X][end]
	U_load = solvers_init[1].stats[:U][end]
	for k = 1:N-1
		x_lift = [x[k] for x in X_lift]
		u_lift = [u[k] for u in U_lift]
		X_batch[k] .= [vcat(x_lift...); X_load[k]]
		U_batch[k] .= [vcat(u_lift...); U_load[k]]
	end
	x_lift = [x[N] for x in X_lift]
	X_batch[N] .= [vcat(x_lift...); X_load[N]]
	J = cost(prob.obj, X_batch, U_batch, dt_traj)
	c_max = max_violation(prob, X_batch, U_batch)
	t_init = maximum([solver.stats[:timestamp][end] for solver in solvers_init])
	push!(Js, J)
	push!(c_maxes, c_max)
	push!(time, t_init)
	push!(labels, :presolve)


	## PARALLEL SOLVE

	# Get sample times
	times = [copy(sol.stats[:timestamp]) for sol in solvers]
	final_time = maximum([t[end] for t in times[2:end]])
	[push!(t, final_time) for t in times[2:end]]
	max_iter = maximum(length.(times[2:end]))
	for i = 2:num_agents
		fillend!(times[i], max_iter)
	end
	# t_sample = unique_approx(vcat(times[2:end]...))
	t_sample = unique(sort(vcat(times[2:end]...)))

	n_load, m_load, N = size(sol[1])
	n_lift, m_lift = size(sol[2])
	n_batch = n_lift*num_lift + n_load
	m_batch = m_lift*num_lift + m_load

	J_total = zeros(length(t_sample))
	c_max_total = zeros(length(t_sample))

	# Get load trajectory from initial solve
	X_load = solvers_init[1].stats[:X][end]
	U_load = solvers_init[1].stats[:U][end]

	for (j,t) in enumerate(t_sample)
		X = [zeros(n_batch) for k = 1:N]
		U = [zeros(m_batch) for k = 1:N-1]

		for k = 1:N
			x_lift = Vector{Float64}[]
			u_lift = Vector{Float64}[]
			for i = 2:num_agents
				Xs = solvers[i].stats[:X]
				Us = solvers[i].stats[:U]
				ts = solvers[i].stats[:timestamp]
				Xi = sample_traj(t, ts, Xs)
				Ui = sample_traj(t, ts, Us)
				push!(x_lift, Xi[k])
				k < N ? push!(u_lift, Ui[k]) : nothing
			end
			X_batch[k] .= [vcat(x_lift...); X_load[k]]
			k < N ? U_batch[k] .= [vcat(u_lift...); U_load[k]] : nothing
		end

		J_total[j] = cost(prob.obj, X_batch, U_batch, dt_traj)
		c_max_total[j] = max_violation(prob, X_batch, U_batch)
	end
	append!(Js, J_total)
	append!(c_maxes, c_max_total)
	append!(time, t_sample)
	append!(labels, repeat([:lift], length(J_total)))

	# Load solve
	append!(Js, solvers[1].stats[:cost_uncon])
	append!(c_maxes, solvers[1].stats[:c_max])
	append!(time, solvers[1].stats[:timestamp])
	append!(labels, repeat([:load], solvers[1].stats[:iterations]))

	return Js, c_maxes, time, labels
end

Js, c_maxes, times, label = recalc_stats(prob, sol_solvers, solvers_init)
Js_b = solver.stats[:cost_uncon]
c_maxes_b = solver.stats[:c_max]
times_b = solver.stats[:timestamp]

function plot_stat!(stat, times, labels)
	inds = findall(label .== :presolve)
	plot!(times[inds], stat[inds], markershape=:circle, color=3, style=:dash, label="Pre-solve (ours)")
	inds = findall(label .== :lift)
	insert!(inds, 1, inds[1]-1)
	plot!(times[inds], stat[inds], markershape=:circle, color=3, style=:solid, label="Quads - parallel (ours)")
	inds = findall(label .== :load)
	insert!(inds, 1, inds[1]-1)
	plot!(times[inds], stat[inds], markershape=:circle, color=3, style=:dot, label="Load (ours)")
end
function plot_stat_pgf!(stat, times, labels)
	inds = findall(label .== :presolve)
	p2 = PGF.Plots.Linear(times[inds], stat[inds], legendentry="Presolve",
		style="color=$col, line width=$lwidth, dashed, mark=*, mark options={$col}",)
	inds = findall(label .== :lift)
	insert!(inds, 1, inds[1]-1)
	p3 = PGF.Plots.Linear(times[inds], stat[inds], legendentry="Quads",
		style="color=$col, line width=$lwidth, solid, mark=*, mark options={$col}",)
	inds = findall(label .== :load)
	insert!(inds, 1, inds[1]-1)
	p4 = PGF.Plots.Linear(times[inds], stat[inds], legendentry="Load",
		style="color=$col, line width=$lwidth, dotted, mark=*, mark options={$col}",)
	return p2, p3, p4
end

using Plots
import Plots.plot
plot(times_b, Js_b, markershape=:circle, label=:Batch, yscale=:log10,
	ylabel=:Cost, xlabel="time (s)")
plot_stat!(Js, times, labels)

plot(times_b, c_maxes_b, markershape=:circle, label=:Batch, yscale=:log10,
	ylabel="Constraint Violation", xlabel="time (s)")
plot_stat!(c_maxes, times, labels)

resetPGFPlotsOptions()
pushPGFPlotsOptions("scale=1.5")
col = "green!70!black"
lwidth = 1.5

p1 = PGF.Plots.Linear(times_b, Js_b, legendentry="Batch", style="line width=$lwidth")
ps = plot_stat_pgf!(Js, times, labels)
a = Axis([p1, ps...],
	legendPos="north east",
	ymode="log",
	hideAxis=false,
	xlabel="time (s)",
	ylabel="cost",
	style="grid=none")
PGF.save("cost_convergence.tikz", a, include_preamble=false)

c1 = PGF.Plots.Linear(times_b, c_maxes_b, legendentry="Batch", style="line width=$lwidth")
cs = plot_stat_pgf!(c_maxes, times, labels)
thres = PGF.Plots.Linear([-1,times_b[end]*1.5], ones(2)*opts_al.constraint_tolerance,
	legendentry="threshold", style="color=red, dashed, no marks")
a = Axis([c1, cs..., thres],
	xmin=-0.5, xmax=times_b[end]*1.1,
	legendPos="north east",
	ymode="log",
	hideAxis=false,
	xlabel="time (s)",
	ylabel="constraint violation",
	style="grid=none")
PGF.save("constraint_convergence.tikz", a, include_preamble=false)
