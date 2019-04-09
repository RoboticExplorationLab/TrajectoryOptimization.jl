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
obj.xf
obj_c = ConstrainedObjective(obj,u_min=0,u_max=4.5)
solver = Solver(model,obj_c,N=N)
res_con,stats_con = solve(solver,U0)

import TrajectoryOptimization: empty_state,num_stage_constraints,num_terminal_constraints, AugmentedLagrangianProblem, AugmentedLagrangianCost
import TrajectoryOptimization: bound_constraint, is_constrained, step!, update_constraints!, dual_update!, penalty_update!
import TrajectoryOptimization: OuterLoop, update_active_set!
costfun = obj.cost
model_d = Model{Discrete}(model,rk4)
U = to_dvecs(U0)
X = empty_state(n,N)
dt = solver.dt
x0 = obj.x0
prob = Problem(model_d,costfun,U0,dt=dt,x0=x0)

# ilqr
opts = iLQRSolverOptions(iterations=50, gradient_norm_tolerance=1e-4, verbose=false)
ilqr = iLQRSolver(prob,opts)
res1 = solve(prob,ilqr)
# plot(res1.U)


# aula
prob = Problem(model_d,costfun,U,dt=dt,x0=x0)
bnd = bound_constraint(n,m,u_min=0,u_max=4.5,trim=true)
add_constraints!(prob,bnd)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=true,unconstrained_solver=opts)
auglag = AugmentedLagrangianSolver(prob,opts_al)
solve!(prob,auglag)
# plot(prob.U)

# altro
prob = Problem(model_d,costfun,U0,dt=dt,x0=x0)
add_constraints!(prob,bnd)

X0 = zeros(prob.model.n,prob.N)
X0[4,:] .= 1.0
X0[3,:] .= range(prob.x0[2],stop=obj.xf[2],length=N)
copyto!(prob.X,X0)

##
# model_altro = prob.model
# con = prob.constraints
# infeasible = true
# min_time = true
prob_altro.model
prob_altro = infeasible_problem(prob,7.5)
prob_altro = minimum_time_problem(prob_altro,3.4)
# prob_altro
#
# cost_altro = copy(prob.cost)
# prob_altro = update_problem(prob,cost=copy(prob.cost))
#
# # if infeasible
# cost_altro.R = cat(cost_altro.R,7.7*Diagonal(I,prob_altro.model.n),dims=(1,2))
# cost_altro.r = [cost_altro.r; zeros(prob_altro.model.n)]
# cost_altro.H = [cost_altro.H; zeros(prob_altro.model.n,prob_altro.model.n)]
# con_inf = infeasible_constraints(model_altro.n,model_altro.m)
# model_altro = add_slack_controls(model_altro)
# u_slack = slack_controls(prob_altro)
#
# prob_altro = update_problem(prob_altro,model=model_altro,cost=cost_altro,
#     constraints=[prob_altro.constraints...,con_inf],U=[[prob_altro.U[k];u_slack[k]] for k = 1:prob_altro.N-1])
# # end
#
# prob_altro
#
# # if min_time
# cost_altro.Q = cat(cost_altro.Q,0,dims=(1,2))
# cost_altro.q = [cost_altro.q; 0]
# cost_altro.R = cat(cost_altro.R,0,dims=(1,2))
# cost_altro.r = [cost_altro.r; 3.4]
# cost_altro.H = [cost_altro.H zeros(prob_altro.model.m,1); zeros(1,prob_altro.model.n+1)]
# cost_altro.Qf = cat(cost_altro.Qf,0,dims=(1,2))
# cost_altro.qf = [cost_altro.qf; 0]
#
# con_min_time_eq, con_min_time_bnd = min_time_constraints(model_altro.n,model_altro.m,1.0,1.0e-3)
# model_altro = add_min_time_controls(model_altro)
#
# prob_altro = update_problem(prob_altro,model=model_altro,cost=cost_altro,
#     constraints=[prob_altro.constraints...,con_min_time_eq,con_min_time_bnd],
#     U=[[prob_altro.U[k];prob_altro.dt] for k = 1:prob_altro.N-1],
#     X=[[prob_altro.X[k];prob_altro.dt] for k = 1:prob_altro.N],
#     x0=[x0;0.0])
# # end
#
# prob_altro
#
# solver_al = AugmentedLagrangianSolver(prob_altro)
# cost_al = AugmentedLagrangianCost(prob_altro,solver_al)
# cost_altro = ALTROCost(prob_altro,cost_al,1.0,1.0)
#
# prob_altro = update_problem(prob_altro,cost=cost_altro)
#
# solver_al.C[1][1]
# cost_al.C[1][1]
# prob_altro.cost.cost.C[1][1]
#
# prob_altro,solver_al = solve!(prob,solver)
#
# reset!(solver_al)
#
# unconstrained_solver = AbstractSolver(prob_altro, solver_al.opts.unconstrained_solver)
# _prob_al = AugmentedLagrangianProblem(prob_altro, solver_al)
#
# J = step!(prob_altro, solver_al, unconstrained_solver)
# prob_altro.cost.cost
#
# unconstrained_solver.
# pro
# _prob_al.cost.C[1][1]

# # solve!(prob_altro,solver_al)
#
# unconstrained_solver = AbstractSolver(prob_altro, solver_al.opts.unconstrained_solver)
# prob_al = AugmentedLagrangianProblem(prob_altro, solver_al)
#
# prob_al.cost.C[1][1]
# solver_al.C[1][1] = 10.0
# prob_al.cost.C[1][1]
# prob_altro.cost.cost.C[1][1]
# J = solve!(prob_altro, unconstrained_solver)
#
# # Outer loop update
# dual_update!(prob_altro, solver_al)
# penalty_update!(prob_altro, solver_al)
# copyto!(solver_al.C_prev,solver_al.C)
#
#
# reset!(solver_al)
#
# unconstrained_solver = AbstractSolver(prob_altro, solver_al.opts.unconstrained_solver)
# prob_al = AugmentedLagrangianProblem(prob_altro, solver_al)
# prob_al
# J = step!(prob_altro, solver_al, prob_al, unconstrained_solver)
#
# solver_al
#
#
#
#
#
#
#
# plot(prob_altro.U)
# prob_altro
# reset!(solver)
#
# unconstrained_solver = AbstractSolver(prob_altro, solver_al.opts.unconstrained_solver)
# prob_al = AugmentedLagrangianProblem(prob_altro, solver_al)
#
# J = step!(prob_altro, solver_al, prob_al, unconstrained_solver)
#
# J = solve!(prob_al, unconstrained_solver)
#
# unconstrained_solver
# jacobian!(prob_al, unconstrained_solver)
#
# Z = rand(13,31)
# prob_al.model.∇f(Z,rand(13),rand(17),1.0)
# unconstrained_solver
#
# ilqr__ = iLQRSolver(prob,iLQRSolverOptions())
# prob.model.∇f(view(Z,1:13,1:18),rand(13),rand(13),rand(4),1.0)
#
#
# model_ = add_slack_controls(prob.model)
# Z = zeros(13,31)
# model_.∇f(Z,x,u,dt)
#
# jacobian!(Z,model_,x,u,1.0)
#
# # n = prob.model.n; m = prob.model.m
# # nm = n+m
# #
# # idx = merge(create_partition((m,n),(:u,:inf)),(x=1:n,))
# # idx2 = [(1:nm)...,2n+m+1]
# #
# # function f!(x₊::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T},dt::T) where T
# #     model.f(x₊,x,u[idx.u],dt)
# #     x₊ .+= u[idx.inf]
# # end
# #
# # function ∇f!(Z::AbstractMatrix{T},x₊::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T},dt::T) where T
# #     prob.model.∇f(view(Z,idx.x,idx2),x₊[idx.x],x[idx.x],u[idx.u],dt)
# #     view(Z,idx.x,(idx.x) .+ nm) .= Diagonal(1.0I,n)
# # end
# #
# # view(Z,idx.x,idx2)
# # x₊ = rand(13)
# # x = rand(13)
# # u = rand(17)
# #
# # x₊[idx.x]
# # x[idx.x]
# # u[idx.u]
# # prob.model.∇f(view(Z,idx.x,idx2),x₊[idx.x],x[idx.x],u[idx.u],dt)
# # prob.model.∇f(view(Z,idx.x,idx2),rand(13),rand(13),rand(4),dt)
# #
# # view(Z,idx.x,(idx.x) .+ nm) .= Diagonal(1.0I,n)
# # Z
# # Z = zeros(13,31)
# # ∇f!(Z,rand(13),rand(13),rand(17),1.0)
# # idx.u
# # altro_cost = ALTROCost(prob,cost_al,NaN,NaN)
# # opts = iLQRSolverOptions(iterations=50, gradient_norm_tolerance=1e-4, verbose=false)
# # cost_expansion!(ilqr.Q,altro_cost,rand(prob.model.n),rand(prob.model.m), 1)
# # auglag = AugmentedLagrangianSolver(prob)
# # prob_al = AugmentedLagrangianProblem(prob,auglag)
# # cost_al = AugmentedLagrangianCost(prob,auglag)
# # cost_altro1 = ALTROCost(prob,cost_al,NaN,NaN)
# # cost(cost_altro1,prob.X,prob.U,prob.dt)
# #
# # # Infeasible
# # copyto!(prob.X,[rand(prob.model.n) for k = 1:prob.N])
# # prob_inf = infeasible_problem(prob)
# #
# # ilqr_inf = iLQRSolver(prob_inf,opts)
#
# cost_expansion!(ilqr_inf.Q,prob_inf.cost,rand(prob_inf.model.n),rand(prob_inf.model.m), 1)
# cost(prob_inf.cost,prob_inf.X,prob_inf.U,prob_inf.dt)
#
# # Minimum time
# prob_min_time = minimum_time_problem(prob)
# ilqr_min_time = iLQRSolver(prob_min_time,opts)
# cost_expansion!(ilqr_min_time.Q,prob_min_time.cost,rand(prob_min_time.model.n),rand(prob_min_time.model.m), 1)
# cost(prob_min_time.cost,prob_min_time.X,prob_min_time.U,prob_min_time.dt)
#
# # Infeasible and Minimum Time
# prob_altro = minimum_time_problem(prob_inf)
# ilqr_altro = iLQRSolver(prob_altro,opts)
# cost_expansion!(ilqr_altro.Q,prob_altro.cost,rand(prob_altro.model.n),rand(prob_altro.model.m), 1)
# cost(prob_altro.cost,prob_altro.X,prob_altro.U,prob_altro.dt)
