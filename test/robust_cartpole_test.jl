using Test, OrdinaryDiffEq, ODEInterface, ODEInterfaceDiffEq
using ODEInterfaceDiffEq

model = Dynamics.pendulum_model_uncertain
n = model.n; m = model.m; r = model.r

T = Float64

# costs
Q = 1.0e-2*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0e-2*Diagonal(I,m)

Qr = 1.0*Diagonal(I,n)
Qfr = 1.0*Diagonal(I,n)
Rr = 1.0*Diagonal(I,m)

x0 = [0.;0.]
xf = [pi;0.]
# x0 = [0.; 0.; 0.; 0.]
# xf = [0.; pi; 0.; 0.]
D = Diagonal((0.2^2)*ones(r))
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,constraint_tolerance=1.0e-3)

N = 101
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:rk3_implicit,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)

initial_controls!(prob,U0)
# rollout!(prob)
solve!(prob,opts_ilqr)
plot(prob.X)
plot(prob.U)


# Kd,Pd = tvlqr_dis(prob,Qr,Rr,Qfr)
# Kdu,Pdu, = tvlqr_dis_uncertain(prob,Qr,Rr,Qfr)
# Kc,Pc,Sc,Ac,Bc,Gc = tvlqr_con_uncertain(prob,Qr,Rr,Qfr,xf,:Tsit5)
#
# Pd_vec = [vec(Pd[k]) for k = 1:N]
# Pdu_vec = [vec(Pdu[k]) for k = 1:N]
# Pc_vec = [vec(Pc[k]) for k = 1:N]
# Sc_vec = [vec(Sc[k]) for k = 1:N]
# plot(Sc_vec)
# S1 = copy(Sc[1])
#
#
# # @test norm(Pd_vec[1] - Pc_vec[1]) < 0.1
# plot(Pd_vec,label="",linetype=:steppost,color=:blue)
# plot(Pdu_vec,label="",linetype=:steppost,color=:green)
# plot!(Pc_vec,label="")

prob_robust = robust_problem(prob,E1,H1,D,Qr,Rr,Qfr,Q,R,Qf,xf)

# rollout!(prob_robust.X,prob_robust.model,prob_robust.U,prob_robust.dt)

xx = [prob_robust.X[k][1:n] for k = 1:N]
ee = [prob_robust.X[k][n .+ (1:n^2)] for k = 1:N]
hh = [prob_robust.X[k][(n+n^2) .+ (1:n*r)] for k = 1:N]
ss = [prob_robust.X[k][(n+n^2+n*r) .+ (1:n^2)] for k = 1:N]

plot(ee)

al_solver = AbstractSolver(prob_robust,opts_al)
ilqr_solver = AbstractSolver(prob_robust,al_solver.opts.opts_uncon)

prob_al = AugmentedLagrangianProblem(prob_robust,al_solver)

J = cost(prob_al.obj,prob_al.X,prob_al.U)
jacobian!(prob_al,ilqr_solver)


nn = prob_robust.model.n; mm = prob_robust.model.m; rr = prob_robust.model.r

ZZ = zeros(nn, nn+mm+rr+1)

prob_robust.model.info[:∇fc](zeros(nn,nn+mm+r),prob_robust.X[1],prob_robust.U[1][1:m],zeros(r))#,prob_robust.dt)
ZZ
