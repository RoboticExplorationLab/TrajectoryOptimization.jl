using Test, OrdinaryDiffEq, ODEInterface, ODEInterfaceDiffEq
using ODEInterfaceDiffEq
using MatrixCalculus

# Model
model = Dynamics.pendulum_model_uncertain
n = model.n; m = model.m; r = model.r
T = Float64

## Costs

# iLQR cost
Q = 1.0e-2*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0e-2*Diagonal(I,m)

# LQR controller
Q_lqr = 1.0*Diagonal(I,n)
Qf_lqr = 10.0*Diagonal(I,n)
R_lqr = 1.0*Diagonal(I,m)

# Robust cost
Q_r = 1.0*Diagonal(I,n)
Qf_r = 1.0*Diagonal(I,n)
R_r = 1.0*Diagonal(I,m)

# initial and final conditions
x0 = [0.;0.]
xf = [pi;0.]
D = Diagonal(0.1*ones(r))
E0 = Diagonal(0.1*ones(n))
H0 = zeros(n,r)

## generate X,U nominal trajectories
costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=2.0,constraint_tolerance=1.0e-3)

N = 101
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:DiffEq_RadauIIA5,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)

initial_controls!(prob,U0)
solve!(prob,opts_ilqr)
plot(prob.X)
plot(prob.U)

prob_robust = robust_problem(prob,E0,H0,D,Q,R,Qf,xf,Q_lqr,R_lqr,Qf_lqr,Q_r,R_r,Qf_r,100.0,1.0)

ee = [prob_robust.X[k][n .+ (1:n^2)] for k = 1:N]
plot(ee)
# solve!(prob_robust,opts_al)

al_solver = AbstractSolver(prob_robust,opts_al)
ilqr_solver = AbstractSolver(prob_robust,al_solver.opts.opts_uncon)
prob_al = AugmentedLagrangianProblem(prob_robust,al_solver)
J = cost(prob_al)
println(J)
jacobian!(prob_al,ilqr_solver)
cost_expansion!(prob_robust,ilqr_solver)

ilqr_solver

ΔV = backwardpass!(prob_al,ilqr_solver)
rollout!(prob_robust,ilqr_solver,0.001)
# forwardpass!(prob_al,ilqr_solver,ΔV,J)

plot(prob.X,labels="")
# plot(prob_robust.X,label="")

xx = [ilqr_solver.X̄[k][1:n] for k = 1:N]
# ee = [ilqr_solver.X̄[k][n .+ (1:n^2)] for k = 1:N]
# plot(ee)
plot!(xx,labels="")

copyto!(prob_robust.X,ilqr_solver.X̄)
copyto!(prob_robust.U,ilqr_solver.Ū)
