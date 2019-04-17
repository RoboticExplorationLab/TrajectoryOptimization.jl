using LinearAlgebra, Random
using Test
using BenchmarkTools
using Plots

model, obj = Dynamics.quadrotor
N = 21
solver = Solver(model,obj,N=N)
n,m = get_sizes(solver)
U0 = 6*ones(m,N-1)
X0 = rollout(solver,U0)
J0 = cost(solver,X0,U0)
res,stats = solve(solver,U0)

obj_c = ConstrainedObjective(obj,u_min=0,u_max=4.5)
solver = Solver(model,obj_c,N=N)
res_con,stats_con = solve(solver,U0)

import TrajectoryOptimization: empty_state,num_stage_constraints,num_terminal_constraints, AugmentedLagrangianProblem, ALCost
import TrajectoryOptimization: bound_constraint, is_constrained, step!, update_constraints!, dual_update!, penalty_update!
import TrajectoryOptimization: OuterLoop, update_active_set!
costfun = obj.cost
model_d = Model{Discrete}(model,rk4)
U = to_dvecs(U0)
X = empty_state(n,N)
dt = solver.dt
x0 = obj.x0
prob = Problem(model_d,costfun,U,dt=dt,x0=x0)
prob
ee = TrajectoryOptimization.Expansion(prob)
eex = Expansion(prob,:x)
opts = iLQRSolverOptions(iterations=50, gradient_norm_tolerance=1e-4, verbose=false)
ilqr = iLQRSolver(prob,opts)
res1 = solve(prob,ilqr)
# plot(res1.X)
rollout!(prob)
@test J0 == cost(prob)
# Constrained
bnd = bound_constraint(n,m,u_min=0,u_max=4.5,trim=true)
add_constraints!(prob,bnd)
@test is_constrained(prob)
res2 = solve(prob,ilqr)
# plot(res2.U)
@test cost(res1) == cost(res2)
Ures = to_array(res2.U)
p = num_stage_constraints(prob)

auglag = AugmentedLagrangianSolver(prob)
prob_al = AugmentedLagrangianProblem(prob,auglag)
cost_al = ALCost(prob,auglag)
cost_al.C[1]
auglag.μ[1][5] = 4.25
@test cost_al.μ[1][5] == 4.25

@test prob_al.cost.μ[1][5] == 4.25
@test !is_constrained(prob_al)
auglag.μ[1][5] = 1
rollout!(prob_al)
cost_al.C[1]
cost(cost_al,prob.X,prob.U,prob.dt) - J0
@test max_violation(auglag) == 1.5
@test max_violation(auglag) == max_violation(prob)

prob = Problem(model_d,costfun,U,dt=dt,x0=x0,N=N)
goal_con = goal_constraint(obj.xf)
con = [bnd, goal_con]
add_constraints!(prob,con)

opts.verbose=true
opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=true,opts_uncon=opts)
# auglag = AugmentedLagrangianSolver(prob,opts_al)
# res3 = solve(prob,auglag)
solve!(prob,opts_al)
