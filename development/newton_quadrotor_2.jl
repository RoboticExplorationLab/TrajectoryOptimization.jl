using Plots
using SparseArrays

# Solver options
N = 51
integration = :rk4
opts = SolverOptions()
opts.verbose = false
opts.square_root = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-2
opts.penalty_max = 1e4

# 3 sphere obstacles + unit quaternion
model, obj_obs = TrajectoryOptimization.Dynamics.quadrotor_3obs

solver = Solver(model,obj_obs,integration=integration,N=N,opts=opts)
U = 0.5*9.81/4.0*ones(solver.model.m, solver.N-1)
results, stats = solve(solver,U)


λ_update_default!(results,solver)
update_constraints!(results,solver)
@assert max_violation(results) < opts.constraint_tolerance

J_prev = cost(solver,results)
c_max_prev = stats["c_max"][end]
J_prev = cost(solver,results)

p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

## Newton 2 ###############
results_new = copy(results)

# newton_solve!(results_new,solver)
newton_results = NewtonResults(solver)
newton_active_set!(newton_results,results_new,solver)
update_newton_results!(newton_results,results_new,solver)
newton_step!(results_new,newton_results,solver,1.0,100.0)
max_violation(results_new)

# get batch problem sizes
Nz = newton_results.Nz
Np = newton_results.Np
Nx = newton_results.Nx

# assemble KKT matrix,vector
_idx1 = Array(1:Nz)
_idx2 = Array((1:Np) .+ Nz)[newton_results.active_set]
_idx3 = Array((1:Nx) .+ (Nz + Np))
_idx4 = Array((1:Np) .+ (Nz + Np + Nx))[newton_results.active_set_ineq]

_idx5 = [_idx1;_idx2;_idx3;_idx4]
eigvals(Array(newton_results.A)[_idx5,_idx5]+1000I)

cond(Array(newton_results.A)[_idx5,_idx5])
cond(Array(newton_results.A)[_idx5,_idx5]+1000I)
