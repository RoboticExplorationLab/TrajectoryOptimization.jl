using Test
using BenchmarkTools
using Plots
using SparseArrays

opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-1
opts.cost_tolerance_intermediate = 1e-1
opts.constraint_tolerance = 1.0
opts.square_root = false
opts.active_constraint_tolerance = 0.0
opts.outer_loop_update_type = :default
opts.live_plotting = false

# Block move
model, obj = TrajectoryOptimization.Dynamics.double_integrator
u_min = -0.2
u_max = 0.2
obj_con = TrajectoryOptimization.ConstrainedObjective(obj, tf=5.0,use_xf_equality_constraint=true, u_min=u_min, u_max=u_max)#, x_min=x_min, x_max=x_max)

solver = TrajectoryOptimization.Solver(model,obj_con,integration=:rk4,N=100,opts=opts)
U = rand(solver.model.m, solver.N)

results, stats = TrajectoryOptimization.solve(solver,U)
@assert max_violation(results) < opts.constraint_tolerance

λ_update_default!(results,solver)

J_prev = cost(solver,results)
c_max_prev = stats["c_max"][end]

# Newton 2
results_new = copy(results)
newton_solve!(results_new,solver)

#
# # newton_solve!(results_new,solver)
# @assert results_new.X == results.X
# @assert results_new.U == results.U
# @assert results_new.C == results.C
# @assert results_new.λ == results.λ
#
# newton_results = NewtonResults(solver)
# newton_active_set!(newton_results,results_new,solver,0.0)
# update_newton_results!(newton_results,results_new,solver)
# newton_step!(results_new,newton_results,solver,0.001)
# c_max = max_violation(results_new)
# J = cost(solver,results_new)
# J-J_prev
# c_max-c_max_prev
#
# cost_constraints(solver,results_new) - cost_constraints(solver,results)
# _cost(solver,results_new) - _cost(solver,results)
#
# @assert results.active_set == results_new.active_set
#
# # project into feasible space
# backwardpass!(results_new,solver)
# rollout!(results_new,solver,1.0)
# results_new.X .= deepcopy(results_new.X_)
# results_new.U .= deepcopy(results_new.U_)
# update_constraints!(results_new,solver)
# JJ = cost(solver,results_new)
# cc_max = max_violation(results_new)
# update_newton_results!(newton_results,results_new,solver)
# norm(newton_results.c)
# norm(vcat(results_new.C...) .+ 0.5*newton_results.s.^2)
# # @assert vcat(results_new.active_set...) == newton_results.active_set
# JJ-J_prev
# cc_max
#
# @assert results_new.active_set == results.active_set
# newton_results.λ[findall(x->x < 0.0, newton_results.λ[newton_results.active_set_ineq])]
#
# 0.5*newton_results.z'*newton_results.∇²J*newton_results.z + newton_results.∇J'*newton_results.z + newton_results.λ'*newton_results.c
