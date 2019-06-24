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

# x0 = [0.;0.]
# xf = [1.0;0.]
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

model_d = discretize_model(model,:midpoint,dt)
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


cost(prob_robust.obj.robust_cost,prob_robust.X,prob_robust.U)
n = prob.model.n; m = prob.model.m; r = prob.model.r
n_robust = 2*n^2 + n*r # number of robust parameters in state vector
n̄ = n + n_robust

_Kc, _Pc, _Sc, _Ac, _Bc, _Gc = tvlqr_con_uncertain(prob,Qr,Rr,Qfr,xf,:Midpoint)
_K, _P, _A, _B, _G = tvlqr_dis_uncertain(prob,Qr,Rr,Qfr)


_Kc_vec = [vec(_Kc[k]) for k = 1:N-1]
_K_vec = [vec(_K[k]) for k = 1:N-1]

plot(_Kc_vec)
plot!(_K_vec)
_S = [vec(cholesky(_P[k]).U) for k = 1:N]
S1 = vec(_S[1])
# S1 = vec(cholesky(_P[1]).U)
SN = vec(cholesky(Qfr).U)

_Z = [zeros(n̄) for k = 1:N]
_E = [zeros(n,n) for k = 1:N]
_H = [zeros(n,r) for k = 1:N]

_E[1] = E1
_H[1] = H1

_Z[1] = [prob.X[1];vec(_E[1]);vec(_H[1]);vec(S1)]

for k = 1:N-1
    # Acl = _A[k] - _B[k]*_K[k]
    Acl = _A[k] - _B[k]*prob_robust.obj.robust_cost.K(_Z[k],prob.U[k])

    _E[k+1] = Acl*_E[k]*Acl' + _G[k]*_H[k]'*Acl' + Acl*_H[k]*_G[k]' + _G[k]*D*_G[k]'
    _H[k+1] = Acl*_H[k] + _G[k]*D

    _Z[k+1] = [prob.X[k+1];vec(_E[k+1]);vec(_H[k+1]);vec(_S[k+1])]
end

ℓ = zeros(1)

for k = 1:N-1
    a = tr((Qr + _K[k]'*Rr*_K[k])*_E[k])
    ℓ[1] += a
    println(a)
end
ℓ[1] += tr(Qfr*_E[N])
ℓ

ℓp = zeros(1)

for k = 1:N
    ℓp[1] += 0.5*vec(_P[k])'*vec(_P[k])
end
ℓp[1]


cost(prob)
# rollout!(prob_robust.X,prob_robust.model,prob_robust.U,prob_robust.dt)

xx = [prob_robust.X[k][1:n] for k = 1:N]
ee = [prob_robust.X[k][n .+ (1:n^2)] for k = 1:N]
hh = [prob_robust.X[k][(n+n^2) .+ (1:n*r)] for k = 1:N]
ss = [prob_robust.X[k][(n+n^2+n*r) .+ (1:n^2)] for k = 1:N]

plot(ee)

tr(Qr*reshape(ee[end],n,n))

al_solver = AbstractSolver(prob_robust,opts_al)
ilqr_solver = AbstractSolver(prob_robust,al_solver.opts.opts_uncon)

prob_al = AugmentedLagrangianProblem(prob_robust,al_solver)

J = cost(prob_al.obj.obj,prob_al.X,prob_al.U)
jacobian!(prob_al,ilqr_solver)
cost_expansion!(prob_al,ilqr_solver)
ΔV = backwardpass!(prob_al,ilqr_solver)

ilqr_solver
prob_al.obj

nn = prob_robust.model.n; mm = prob_robust.model.m; rr = prob_robust.model.r

ZZ = zeros(nn, nn+mm+rr+1)

prob_robust.model.info[:∇fc](zeros(nn,nn+mm+r),prob_robust.X[1],prob_robust.U[1][1:m],zeros(r))#,prob_robust.dt)
prob_robust.model.∇f(ZZ,prob_robust.X[3],prob_robust.U[3][1:m],zeros(r),prob_robust.dt)

ZZ
