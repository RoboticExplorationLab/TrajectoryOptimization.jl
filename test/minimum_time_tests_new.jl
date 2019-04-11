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
opts_ilqr = iLQRSolverOptions{T}(verbose=true)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=true,opts_uncon=opts_ilqr,iterations=50,penalty_scaling=2.0)
opts_altro = ALTROSolverOptions{T}(verbose=true,opts_con=opts_al,R_minimum_time=15.0,dt_max=0.15,dt_min=1.0e-3)

# constraints
u_bnd = 5.
bnd = bound_constraint(n,m,u_min=-u_bnd,u_max=u_bnd,trim=true)

# problem
N = 31
U = [ones(m) for k = 1:N-1]
dt = 0.15/2.0
prob = Problem(model_d,lqr_cost,U,dt=dt,x0=x0,tf=:min)
add_constraints!(prob,bnd)

# prob = minimum_time_problem(prob,15.0)


solve!(prob,opts_altro)

using Plots
plot(prob.U)

## altro breakdown

# prob_altro = prob
# opts = opts_altro
# if prob_altro.tf == 0.0
#     println("Minimum Time Solve")
#     prob_altro = minimum_time_problem(prob_altro,opts.R_minimum_time,
#         opts.dt_max,opts.dt_min)
# end
# prob_altro
# solver_al = AbstractSolver(prob_altro,opts.opts_con)
#
# unconstrained_solver = AbstractSolver(prob_altro, solver_al.opts.opts_uncon)
#
# prob_al = AugmentedLagrangianProblem(prob_altro, solver_al)
#
# prob_al.cost.C[1][:min_time_eq]
