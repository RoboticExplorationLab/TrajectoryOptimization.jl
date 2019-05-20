model = Dynamics.pendulum_model_uncertain
n = model.n; m = model.m; r = model.r

T = Float64
# costs
Q = 1.0*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0*Diagonal(I,m)
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)

D = (0.2^2)*ones(r)
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-6)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-4)

N = 501
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:midpoint_implicit,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)
initial_controls!(prob,U0)
rollout!(prob)
plot(prob.X)
Kd,Pd = tvlqr_dis(prob,Q,R,Qf)
Kc,Pc = tvlqr_con(prob,Q,R,Qf,xf)
#
Pd_vec = [vec(Pd[k]) for k = 1:N]
Pc_vec = [vec(Pc[k]) for k = 1:N]

plot(Pd_vec,label="",linetype=:steppost)
plot!(Pc_vec,label="")

prob_robust = robust_problem(prob,E1,H1,D,Q,R,Qf,Q,R,Qf,xf)
initial_controls!(prob_robust,U0)

rollout!(prob_robust)
prob_robust.X[1]
prob_robust.U[1]

ilqr_solver = AbstractSolver(prob_robust,iLQRSolverOptions())

jacobian!(prob_robust,ilqr_solver)

backwardpass!(prob_robust,ilqr_solver)
forwardpass!(prob_robust,ilqr_solver,[0.;0.],1e8)

prob_robust.obj
