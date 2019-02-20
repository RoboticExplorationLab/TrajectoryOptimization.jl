using Test
using BenchmarkTools
using Plots
using SparseArrays

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-5
opts.square_root = false
opts.active_constraint_tolerance = 0.0
opts.outer_loop_update_type = :default
opts.penalty_max = 1e8
opts.live_plotting = false

# Parallel Park
model, = TrajectoryOptimization.Dynamics.dubinscar_parallelpark
# initial and goal states
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

# costs
Q = (1e-2)*Diagonal(I,model.n)
Qf = 100.0*Diagonal(I,model.n)
R = (1e-2)*Diagonal(I,model.m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
obj = LQRObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)


solver = TrajectoryOptimization.Solver(model,obj_con,integration=:rk4,N=31,opts=opts)
U = rand(solver.model.m, solver.N)

results, stats = TrajectoryOptimization.solve(solver,U)
λ_update_default!(results,solver)
update_constraints!(results,solver)
@assert max_violation(results) < opts.constraint_tolerance

J_prev = cost(solver,results)
c_max_prev = stats["c_max"][end]

p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)
## Newton 2 ###############
results_new = copy(results)

# newton_solve!(results_new,solver)

newton_results = NewtonResults(solver)
newton_active_set!(newton_results,results_new,solver)
# sum(newton_results.active_set)
# sum(vcat(results_new.active_set...))
# sum(newton_results.active_set_ineq)
# newton_results.s[findall(x->x != 0.0, newton_results.s)]
# findall(x->x < 0.0, vcat(results.C...)[newton_results.active_set_ineq])
update_newton_results!(newton_results,results_new,solver)
newton_step!(results_new,newton_results,solver,1.0)
max_violation(results_new)


a = 1
# # project into feasible space
# backwardpass!(results_new,solver)
# rollout!(results_new,solver,1.0)
# results_new.X .= deepcopy(results_new.X_)
# results_new.U .= deepcopy(results_new.U_)
# update_constraints!(results_new,solver)
# cost(solver,results_new)
# max_violation(results_new)
#
# newton_results.λ[findall(x->x < 0.0, newton_results.λ[newton_results.active_set_ineq])]
#
# a = 1
# results_new = copy(results)
# λ_update_default!(results_new,solver)
#
# newton_results = NewtonResults(solver)
# newton_active_set!(newton_results,results_new,solver)
# update_newton_results!(newton_results,results_new,solver)
# newton_step!(results_new,newton_results,solver,1.0)
# max_violation(results_new)
#
# plot(results_new.X)
# plot!(results.X)
# plot(results_new.U)
# plot!(results.U)
# ##########################
# backwardpass!(results_new,solver)
# rollout!(results_new,solver,1.0)
# results_new.X .= deepcopy(results_new.X_)
# results_new.U .= deepcopy(results_new.U_)
# update_constraints!(results_new,solver)
# cost(solver,results_new)
# max_violation(results_new)
# #########################
#
# results_new_new = copy(results_new)
# newton_results_new = copy(newton_results)
# max_violation(results_new_new)
# @assert newton_results_new.active_set == newton_results.active_set
# @assert results_new_new.X == results_new.X
# @assert results_new_new.U == results_new.U
# @assert results_new_new.C == results_new.C
# update_constraints!(results_new_new,solver)
# newton_active_set!(newton_results_new,results_new_new,solver)
# update_newton_results!(newton_results_new,results_new_new,solver)
# newton_step!(results_new_new,newton_results_new,solver,1.0)
# max_violation(results_new_new)
#
#
# backwardpass!(results_new_new,solver)
# results_new_new.X .= deepcopy(results_new_new.X_)
# results_new_new.U .= deepcopy(results_new_new.U_)
# update_constraints!(results_new_new,solver)
# cost(solver,results_new_new)
# max_violation(results_new_new)
#
# @assert results_new_new.X == results.X
# @assert results_new_new.X == results_new.X
#
#
#

a = 1

#
# # copy results for safe keeping
# results_new = copy(results)
#
# # do a final dual update
# λ_update_default!(results_new,solver)
#
# # initialize newton results
# newton_results = NewtonResults(solver)
#
# # evaluate ϵ-active set
# newton_active_set!(newton_results,results_new,solver)
#
# # move info from results to newton results
# update_newton_results!(newton_results,results_new,solver)
#
# # solve δ = A\b
# solve_kkt!(newton_results)
#
# newton_results_cache = copy(newton_results)
#
# newton_results = copy(newton_results_cache)
# # α-step
# α = 1.0
# _idx1 = Array(1:newton_results.Nz)
# _idx2 = Array((1:newton_results.Np) .+ newton_results.Nz)[newton_results.active_set]
# _idx3 = Array((1:newton_results.Nx) .+ (newton_results.Nz + newton_results.Np))
# _idx4 = Array((1:newton_results.Np) .+ (newton_results.Nz + newton_results.Np + newton_results.Nx))[newton_results.active_set_ineq]
#
# newton_results.z .+= α*newton_results.δ[_idx1]
# newton_results.λ[newton_results.active_set] += α*newton_results.δ[_idx2]
# newton_results.ν .+= α*newton_results.δ[_idx3]
# newton_results.s[newton_results.active_set_ineq] += α*newton_results.δ[_idx4]
#
# @assert newton_results.s != newton_results_cache.s
# @assert newton_results.z != newton_results_cache.z
# @assert newton_results.λ != newton_results_cache.λ
#
# update_results_from_newton_results!(results_new,newton_results,solver)
#
# backwardpass!(results_new,solver)
#
# α = 1.0
# rollout!(results_new,solver,α)
# plot(results_new.U)
# plot!(results.U)
# plot(results_new.X)
# plot!(results.X)
# cost(solver, results_new, results_new.X, results_new.U)
# max_violation(results_new)
#
# results_new.d
# results.d
# newton_active_set!(newton_results,results_new,solver)
#
# # check for max constraint decrease
# max_violation(results_new)
#
# ## check KKT conditions
# newton_results.∇J + newton_results.∇c'*newton_results.λ + newton_results.∇d'*newton_results.ν
# norm(0.5*newton_results.s.^2 .+ vcat(results_new.C...))
# newton_results.d
# newton_results.λ .* newton_results.s
