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
goal_con = goal_constraint(xf)

con = [bnd, goal_con]

# problem
N = 31
U = [ones(m) for k = 1:N-1]
dt = 0.15/2.0
prob = Problem(model_d,lqr_cost,U,dt=dt,x0=x0)
add_constraints!(prob,con)
# X0 = zeros(prob.model.n,prob.N)
# X0[1,:] .= range(prob.x0[1],stop=xf[1],length=N)
# copyto!(prob.X,X0)
using Plots
plot(prob.X)
# prob_inf = infeasible_problem(prob,opts_altro.R_inf)
prob
solve!(prob,opts_altro)

using Plots
plot(prob.U)
