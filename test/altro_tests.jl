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
res_con.U

import TrajectoryOptimization: empty_state,num_stage_constraints,num_terminal_constraints, AugmentedLagrangianProblem, AugmentedLagrangianCost
import TrajectoryOptimization: bound_constraint, is_constrained, step!, update_constraints!, dual_update!, penalty_update!
import TrajectoryOptimization: OuterLoop, update_active_set!
costfun = obj.cost
model_d = Model{Discrete}(model,rk4)
U = to_dvecs(U0)
X = empty_state(n,N)
dt = solver.dt
x0 = obj.x0
prob = Problem(model_d,costfun,U,dt=dt,x0=x0)

# ilqr
opts = iLQRSolverOptions(iterations=50, gradient_norm_tolerance=1e-4, verbose=false)
ilqr = iLQRSolver(prob,opts)
res1 = solve(prob,ilqr)

# aula
bnd = bound_constraint(n,m,u_min=0,u_max=4.5,trim=true)
add_constraints!(prob,bnd)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=true,unconstrained_solver=opts)
auglag = AugmentedLagrangianSolver(prob,opts_al)
solve!(prob,auglag)


# altro
prob = Problem(model_d,costfun,U,dt=dt,x0=x0)
add_constraints!(prob,bnd)
T = Float64
opts = ALTROSolverOptions{T}()
solver = AbstractSolver(prob,opts)
solver.opts.solver_al
X0 = zeros(prob.model.n,prob.N)
X0[4,:] .= 1.0
X0[3,:] .= range(prob.x0[2],stop=obj.xf[2],length=N)
copyto!(prob.X,X0)
prob_altro = infeasible_problem(prob,solver.opts.R_inf)

solver_al = AbstractSolver(prob_altro,solver.opts.solver_al)

solve!(prob_altro,solver_al)

reset!(solver)

unconstrained_solver = AbstractSolver(prob_altro, solver_al.opts.unconstrained_solver)
prob_al = AugmentedLagrangianProblem(prob_altro, solver_al)

J = step!(prob_altro, solver_al, prob_al, unconstrained_solver)

J = solve!(prob_al, unconstrained_solver)

unconstrained_solver
jacobian!(prob_al, unconstrained_solver)

Z = rand(13,31)
prob_al.model.∇f(Z,rand(13),rand(17),1.0)
unconstrained_solver

ilqr__ = iLQRSolver(prob,iLQRSolverOptions())
prob.model.∇f(view(Z,1:13,1:18),rand(13),rand(13),rand(4),1.0)


model_ = add_slack_controls(prob.model)
Z = zeros(13,31)
model_.∇f(Z,x,u,dt)

jacobian!(Z,model_,x,u,1.0)

# n = prob.model.n; m = prob.model.m
# nm = n+m
#
# idx = merge(create_partition((m,n),(:u,:inf)),(x=1:n,))
# idx2 = [(1:nm)...,2n+m+1]
#
# function f!(x₊::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T},dt::T) where T
#     model.f(x₊,x,u[idx.u],dt)
#     x₊ .+= u[idx.inf]
# end
#
# function ∇f!(Z::AbstractMatrix{T},x₊::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T},dt::T) where T
#     prob.model.∇f(view(Z,idx.x,idx2),x₊[idx.x],x[idx.x],u[idx.u],dt)
#     view(Z,idx.x,(idx.x) .+ nm) .= Diagonal(1.0I,n)
# end
#
# view(Z,idx.x,idx2)
# x₊ = rand(13)
# x = rand(13)
# u = rand(17)
#
# x₊[idx.x]
# x[idx.x]
# u[idx.u]
# prob.model.∇f(view(Z,idx.x,idx2),x₊[idx.x],x[idx.x],u[idx.u],dt)
# prob.model.∇f(view(Z,idx.x,idx2),rand(13),rand(13),rand(4),dt)
#
# view(Z,idx.x,(idx.x) .+ nm) .= Diagonal(1.0I,n)
# Z
# Z = zeros(13,31)
# ∇f!(Z,rand(13),rand(13),rand(17),1.0)
# idx.u
# altro_cost = ALTROCost(prob,cost_al,NaN,NaN)
# opts = iLQRSolverOptions(iterations=50, gradient_norm_tolerance=1e-4, verbose=false)
# cost_expansion!(ilqr.Q,altro_cost,rand(prob.model.n),rand(prob.model.m), 1)
# auglag = AugmentedLagrangianSolver(prob)
# prob_al = AugmentedLagrangianProblem(prob,auglag)
# cost_al = AugmentedLagrangianCost(prob,auglag)
# cost_altro1 = ALTROCost(prob,cost_al,NaN,NaN)
# cost(cost_altro1,prob.X,prob.U,prob.dt)
#
# # Infeasible
# copyto!(prob.X,[rand(prob.model.n) for k = 1:prob.N])
# prob_inf = infeasible_problem(prob)
#
# ilqr_inf = iLQRSolver(prob_inf,opts)
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
