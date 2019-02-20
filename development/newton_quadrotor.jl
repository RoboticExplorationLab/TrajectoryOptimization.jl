using Test
using BenchmarkTools
using Plots
using SparseArrays

model,obj = Dynamics.quadrotor
n = 13
m = 4

# Unconstrained
Q = (1e-3)*Matrix(I,n,n)
R = (1e-2)*Matrix(I,m,m)
# R = (1e-1)*Matrix(I,m,m)
Qf = (1000.0)*Matrix(I,n,n)
tf = 1.0
dt = 0.1

# -initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; 0.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;10.;0.] # xyz position
xf[4:7] = q0
u_max = 10.0
u_min = 0.0

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max)#,u_min=u_min)
solver = Solver(model,obj_con,N=50)

solver.obj
solver.opts.cost_tolerance = 1e-5
solver.opts.constraint_tolerance = 1e-4
solver.opts.penalty_max = 1e4
n,m,N = get_sizes(solver)
Random.seed!(1)
U = 0.5*9.81/4.0*ones(solver.model.m, solver.N-1)
res,stats = solve(solver,U)
plot(res.U)
plot(res.X)
cost(solver,res)
λ_update_default!(res,solver);
update_constraints!(res,solver)
max_violation(res)
@assert max_violation(res) < opts.constraint_tolerance

J_prev = cost(solver,res)

p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

## Newton 2 ###############
results_new = copy(res)
newton_results = NewtonResults(solver)
newton_active_set!(newton_results,results_new,solver)
update_newton_results!(newton_results,results_new,solver)
newton_step!(results_new,newton_results,solver,1.0)
max_violation(results_new)
newton_cost(res,newton_results,solver)
