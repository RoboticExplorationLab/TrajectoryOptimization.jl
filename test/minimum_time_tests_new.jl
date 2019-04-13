# Pendulum
T = Float64

# model
dyn = TrajectoryOptimization.Dynamics.pendulum_dynamics!
n = 2; m = 1
model = Model(dyn,n,1)
model_d = Model{Discrete}(model,rk4)

# cost
Q = Array(1e-3*Diagonal(I,n))
R = Array(1e-3*Diagonal(I,m))
Qf = Array(Diagonal(I,n)*0.0)
x0 = zeros(n)
xf = [pi;0.0]
lqr_cost = LQRCost(Q,R,Qf,xf)

# options
opts_ilqr = iLQRSolverOptions{T}(verbose=true,live_plotting=:control)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=true,opts_uncon=opts_ilqr,iterations=50,penalty_scaling=2.0)
opts_altro = ALTROSolverOptions{T}(verbose=true,opts_al=opts_al,R_minimum_time=15.0,dt_max=0.15,dt_min=1.0e-3)

# constraints
u_bnd = 5.
bnd = bound_constraint(n,m,u_min=-u_bnd,u_max=u_bnd,trim=true)
bnd
goal_con = goal_constraint(xf)

con = [bnd, goal_con]
xf
# problem
N = 31
U = [ones(m) for k = 1:N-1]
dt = 0.15/2.0
prob = Problem(model_d,lqr_cost,U,dt=dt,x0=x0,tf=:min)
add_constraints!(prob,con)
# # X0 = zeros(prob.model.n,prob.N)
# # X0[1,:] .= range(prob.x0[1],stop=xf[1],length=N)
# # copyto!(prob.X,X0)
# m
# prob_mt = minimum_time_problem(prob,15.0,0.15,1.0e-3)
# # update_constraint_set_jacobians(prob_mt.constraints,n,n+1,m)
# _con = update_constraint_set_jacobians(prob_mt.constraints,n,n+1,m)
# copy(_con)
#
# prob_mt = update_problem(prob_mt,constraints=_con)
# rollout!(prob_mt)
# rollout!(prob)
#
# plot(prob_mt.X)
# plot!(prob.X)
#
# cost(prob_mt)
# ilqr_solver = AbstractSolver(prob_mt,opts_ilqr)
# ilqr_solver2 = AbstractSolver(prob,opts_ilqr)
# al_solver = AbstractSolver(prob_mt,opts_al)
# prob_al = AugmentedLagrangianProblem(prob_mt,al_solver)
#
# # update_constraints!(prob_al.cost,prob_al.X,prob_al.U)
# k = 3
# jacobian!(prob_al.cost.cost,prob_al.cost.∇C[k],prob_al.cost.constraints,prob_al.X[k],prob_al.U[k],k)
# jacobian!(prob_al.cost.∇C[end],prob_al.cost.constraints,prob_al.X[end])
#
# jacobian!(prob_al,ilqr_solver)
# jacobian!(prob,ilqr_solver2)
#
# prob_al.dt
# prob_al.dt
#
# plot(prob.X)
# plot(prob_al.X)
# ilqr_solver.∇F[end]
#
#
# prob_al.cost.constraints
# ilqr_solver2.∇F[end]
#
# prob_al.cost.∇C[k][:bound]
# prob_al.cost.∇C[end]
# f(t) = 1+t
# type(con)
# con = prob_al.cost.constraints[1]
# con.c
# con.∇c
# n
# m
# con.p
# con.label
# con.inds
# f(a,x,u) = 1.0
# g(a,x,u) = 2.0
# Constraint{type(con)}(con.c,con.∇c, con.p, con.label, con.inds)
#
# function update_constraint_set_jacobians(cs::AbstractConstraintSet,n::Int,n̄::Int,m::Int)
#     idx = [(1:n)...,((1:m) .+ n̄)...]
#     _cs = []
#
#     for con in stage(cs)
#         _∇c(C,x,u) = con.∇c(view(C,:,idx),x,u)
#         push!(_cs,Constraint{type(con)}(con.c,_∇c,n,m,con.p,con.label,inds=con.inds))
#     end
#
#     for con in terminal(cs)
#         push!(_cs,con)
#     end
#
#     return [_cs...]
# end
#
# function update_constraint_set_jacobians!(cs::AbstractConstraintSet,n::Int,n̄::Int,m::Int)
#     idx = [(1:n)...,((1:m) .+ n̄)...]
#
#     for con in stage(cs)
#         _∇c(C,x,u) = con.∇c(view(C,:,idx),x,u)
#         con.∇c .= _∇c
#         # con = Constraint{type(con)}(con.c,_∇c,n,m,con.p,con.label,inds=con.inds)
#     end
#     return nothing
# end
#
# for c in prob_al.cost.constraints
#     if c.label == :bound
#         print("hi")
#         c.inds[2] .+= 1
#     end
# end
#
# prob_al.cost.constraints[1]
# prob
solve!(prob,opts_altro)

# using Plots
# plot(prob.U)
