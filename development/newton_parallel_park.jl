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
