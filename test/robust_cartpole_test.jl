using Test

model = Dynamics.cartpole_model_uncertain
n = model.n; m = model.m; r = model.r

T = Float64

# costs
Q = 1.0e-2*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0e-2*Diagonal(I,m)

Qr = 1.0*Diagonal(I,n)
Qfr = 1.0*Diagonal(I,n)
Rr = 1.0*Diagonal(I,m)

x0 = [0.; 0.; 0.; 0.]
xf = [pi/2; 0.; 0.; 0.]
D = Diagonal((0.2^2)*ones(r))
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,constraint_tolerance=1.0e-3)

N = 201
tf = 5.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:rk3_implicit,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)

initial_controls!(prob,U0)
# rollout!(prob)
solve!(prob,opts_ilqr)
plot(prob.X)
plot(prob.U)
Kd,Pd = tvlqr_dis(prob,Qr,Rr,Qfr)
Kc,Sc = tvlqr_sqrt_con_rk3_uncertain(prob,Qr,Rr,Qfr,xf)
#
Pd_vec = [vec(Pd[k]) for k = 1:N]
Pc_vec = [vec(reshape(Sc[k],n,n)*reshape(Sc[k],n,n)') for k = 1:N]
S1 = copy(Sc[1])

# @test norm(Pd_vec[1] - Pc_vec[1]) < 0.1
plot(Pd_vec,label="",linetype=:steppost)
plot!(Pc_vec,label="")

prob_robust = robust_problem(prob,E1,H1,D,Qr,Rr,Qfr,Q,R,Qf,xf)

rollout!(prob_robust)

# jacobian!(prob_robust,ilqr_solver)
# cost_expansion!(prob_robust,ilqr_solver)
# ΔV = backwardpass!(prob_robust,ilqr_solver)
#
# ilqr_solver
# rollout!(prob_robust,ilqr_solver,0.01)
#
# plot(prob_robust.X)
# plot!(ilqr_solver.X̄)
# _J = cost(prob)
# forwardpass!(prob_robust,ilqr_solver,ΔV,J)
# prob_robust.X

opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:state)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,penalty_initial=10.,constraint_tolerance=1.0e-3)

al_solver = AbstractSolver(prob_robust,opts_al)

solve!(prob_robust,al_solver)


xx = [prob_robust.X[k][1:n] for k = 1:N]
uu = [prob_robust.U[k][1:m] for k = 1:N-1]

p = plot(prob.X,title="Pendulum State",label="nominal",color=:orange,width=2)
p = plot!(xx,color=:blue,label="robust",width=2,legend=:left)

DIR = joinpath(TrajectoryOptimization.root_dir(),"results")
savefig(joinpath(DIR,"pendulum_state.png"))

p = plot(prob.U,title="Pendulum Control",label="nominal",color=:orange,width=2)
p = plot!(uu,color=:blue,width=2,label="robust",legend=:left)

DIR = joinpath(TrajectoryOptimization.root_dir(),"results")
savefig(joinpath(DIR,"pendulum_control.png"))

# max_violation(al_solver)
#
# prob_al
#
# plot(prob_al.X)
#
#
# plot(prob_robust.X)
# plot!(prob_al.X)
# solve!(prob_robust,opts_al)
