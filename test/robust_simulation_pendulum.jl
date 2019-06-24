using Test, OrdinaryDiffEq, ODEInterface, ODEInterfaceDiffEq
using ODEInterfaceDiffEq
using MatrixCalculus

model = Dynamics.pendulum_model_uncertain
n = model.n; m = model.m; r = model.r

T = Float64

# costs
Q = 1.0*Diagonal(I,n)
Qf = 10.0*Diagonal(I,n)
R = 1.0*Diagonal(I,m)

Qr = 1.0*Diagonal(I,n)
Qfr = 1.0*Diagonal(I,n)
Rr = 1.0*Diagonal(I,m)

x0 = [0.;0.]
xf = [pi;0.]
D = Diagonal((0.2^2)*ones(r))
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

## X,U nominal
costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,constraint_tolerance=1.0e-3)

N = 201
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:DiffEq_Tsit5,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)

initial_controls!(prob,U0)
solve!(prob,opts_ilqr)
plot(prob.X)
plot(prob.U)
#
# # P/S nominal
# _Kc, _Pc, _Sc, _Ac, _Bc, _Gc = tvlqr_con_uncertain(prob,Qr,Rr,Qfr,xf,:RadauIIA5)
# _Kd, _Pd, _Ad, _Bd, _Gd = tvlqr_dis_uncertain(prob,Qr,Rr,Qfr)
#
# _Sc_vec = [vec(_Sc[k]) for k = 1:N]
#
# _Pc_vec = [vec(_Pc[k]) for k = 1:N]
# _Pd_vec = [vec(_Pd[k]) for k = 1:N]
# __Pc_vec = [vec(_Sc[k]*_Sc[k]') for k = 1:N]
#
# _Kc_vec = [vec(_Kc[k]) for k = 1:N-1]
# _Kd_vec = [vec(_Kd[k]) for k = 1:N-1]
#
# plot(_Pc_vec,color=:orange,labels="")
# plot!(_Pd_vec,color=:blue,labels="")
# plot!(__Pc_vec,color=:green,labels="")
#
# plot(_Kc_vec,color=:orange,labels="")
# plot!(_Kd_vec,color=:blue,labels="")
#
# # E,H nominal
# n = prob.model.n; m = prob.model.m; r = prob.model.r
# n_robust = 2*n^2 + n*r # number of robust parameters in state vector
# n̄ = n + n_robust
# _Z = [zeros(n̄) for k = 1:N]
# _E = [zeros(n,n) for k = 1:N]
# _H = [zeros(n,r) for k = 1:N]
#
# _E[1] = E1
# _H[1] = H1
#
# _Z[1] = [prob.X[1];vec(_E[1]);vec(_H[1]);vec(_Sc[1])]
#
# _A = _Ad; _B = _Bd; _G = _Gd; _K = _Kd; _S = _Sc
# for k = 1:N-1
#     Acl = _A[k] - _B[k]*_K[k]
#     _E[k+1] = Acl*_E[k]*Acl' + _G[k]*_H[k]'*Acl' + Acl*_H[k]*_G[k]' + _G[k]*D*_G[k]'
#     _H[k+1] = Acl*_H[k] + _G[k]*D
#
#     _Z[k+1] = [prob.X[k+1];vec(_E[k+1]);vec(_H[k+1]);vec(_S[k+1])]
# end
#
# _E_vec = [vec(_E[k]) for k = 1:N]
# _H_vec = [vec(_H[k]) for k = 1:N]
#
# plot(_E_vec)
# plot(_H_vec)
# plot(_Sc_vec)
#
# plot(_Z,labels="")
#
# model_robust = robust_model(model.f,Qr,Rr,D,n,m,r)
#
# model_robust_d = discretize_model(model_robust,:DiffEq_RadauIIA5,prob.dt)
#
# Zd = zeros(model_robust_d.n,model_robust_d.n+model_robust_d.m+model_robust.r + 1)
#
# model_robust_d.∇f(Zd,_Z[2],prob.U[2],zeros(r),dt)
#
#
# J = cost(prob)
#
# J_robust = zeros(1)
#
# for k = 1:N-1
#     J_robust[1] += tr((Qr + _Kc[k]'*Rr*_Kc[k])*_E[k])# + 0.5*vec(_Sc[k])'*vec(_Sc[k])*0.1
# end
# J_robust[1] += tr(Qfr*_E[N])
#
# J_robust


prob_robust = robust_problem(prob,E1,H1,D,Qr,Rr,Qfr,Q,R,Qf,xf)

solve!(prob_robust,opts_al)

# # xx = [prob_robust.X[k][1:n] for k = 1:N]
# # ee = [prob_robust.X[k][n .+ (1:n^2)] for k = 1:N]
# # hh = [prob_robust.X[k][(n+n^2) .+ (1:n*r)] for k = 1:N]
# # ss = [prob_robust.X[k][(n+n^2+n*r) .+ (1:n^2)] for k = 1:N]
# #
# # plot(xx)
# # plot(ee)
# # plot(hh)
# # plot(ss)
#
# cost(prob_robust.obj.robust_cost,prob_robust.X,prob_robust.U)
#
# prob_robust.obj.robust_cost.K(prob_robust.X[2],prob_robust.U[2])
# #
# # _Kd[2]
# # prob.model.f(rand(n),rand(n),rand(m),rand(r),1.4)
# # _f(x⁺,z) = prob.model.f(x⁺,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r),z[n+m+1])
# # _f(rand(n),rand(n+m+1))
# # _∇f(z) = ForwardDiff.jacobian(_f,zeros(eltype(z),n),z)
# # _∇f(rand(n+m+1))
# # _∇f(x,u) = _∇f([x;u;prob.dt])
# # _∇f(rand(n+m+r))
# #
# idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)),z=(1:n̄))
# #
# # function __K(z,u)
# #     x = z[idx.x]
# #     s = z[idx.s]
# #     P = reshape(s,n,n)*reshape(s,n,n)'
# #     Fd = _∇f(x,u)
# #     Ad = Fd[:,1:n]
# #     Bd = Fd[:,n .+ (1:m)]
# #     println(Ad)
# #     println(Bd)
# #     println(P)
# #     (Rr*prob.dt + Bd'*P*Bd)\(Bd'*P*Ad)
# # end
# f(ẋ,z) = prob.model.info[:fc](ẋ,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r))
# ∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
# ∇f(x,u) = ∇f([x;u])
#
# function K(z,u)
#     x = z[idx.x]
#     s = z[idx.s]
#     P = reshape(s,n,n)*reshape(s,n,n)'
#     Bc = ∇f(x,u)[:,n .+ (1:m)]
#     Rr\(Bc'*P)
# end
#
# K(prob_robust.X[10],prob_robust.U[10])
# _Kc[10]
#
#
#
# Zd = zeros(model_robust_d.n,model_robust_d.n+model_robust_d.m+model_robust.r + 1)
# Zdd = zeros(model_robust_d.n,model_robust_d.n+model_robust_d.m+model_robust.r + 1)
#
# model_robust_d.∇f(Zd,_Z[2],prob.U[2],zeros(r),dt)
# prob_robust.model.∇f(Zdd,_Z[2],prob.U[2],zeros(r),dt)
# prob_robust.model
# prob_robust.model.∇f(Zdd,prob_robust.X[2],prob_robust.U[2],zeros(r),dt)
# Zdd
#
# model_robust_d
# prob_robust.model
#
al_solver = AbstractSolver(prob_robust,opts_al)
ilqr_solver = AbstractSolver(prob_robust,al_solver.opts.opts_uncon)
prob_al = AugmentedLagrangianProblem(prob_robust,al_solver)
J = cost(prob_al)
jacobian!(prob_al,ilqr_solver)
cost_expansion!(prob_robust,ilqr_solver)


ΔV = backwardpass!(prob_al,ilqr_solver)
rollout!(prob_robust,ilqr_solver,0.001)
forwardpass!(prob_al,ilqr_solver,ΔV,J)


plot(prob_robust.X,label="")
plot!(ilqr_solver.X̄,label="")
copyto!(prob_robust.X,ilqr_solver.X̄)
copyto!(prob_robust.U,ilqr_solver.Ū)

∇sc, ∇²sc, ∇sc_term, ∇²sc_term = gen_robust_exp_funcs(K,idx,Qr,Rr,Qfr,n,m,r)

dJ_fd = ∇sc([_Z[2];prob.U[2]])
dJN_fd = ∇sc_term(_Z[N])
∇²sc([_Z[2];prob.U[2]])


k = 2
dJdK = reshape((Rr)*_Kc[k]*_E[k] + (Rr)'*_Kc[k]*_E[k]',1,m*n)
dJdE = vec((Qr + _Kc[k]'*(Rr)*_Kc[k])')
dKdB = kron(_Pc[k]',inv(Rr))*comm(n,m)
dKdP = kron(Diagonal(ones(n)),(Rr)\_Bc[k]')
dKdS = kron(_Sc[k],(Rr)\_Bc[k]') + kron(Diagonal(ones(n)), (Rr)\(_Bc[k]'*_Sc[k]))*comm(n,n)

function Bc(x,u)
    ∇f(x,u)[:,n .+ (1:m)]
end

Bc(z) = Bc(z[1:n],z[n .+ (1:m)])
∇Bc(z) = ForwardDiff.jacobian(Bc,z)
Bc([prob.X[2];prob.U[2]])
∇Bc([prob.X[2];prob.U[2]])

dBdX = ∇Bc([prob.X[k];prob.U[k]])[:,1:n]
dBdU = ∇Bc([prob.X[k];prob.U[k]])[:,n .+ (1:m)]

dJ = zeros(n̄+m)
dJ[idx.x] = dJdK*dKdB*dBdX
dJ[idx.e] = dJdE
dJ[idx.s] = dJdK*dKdS
dJ[n̄ .+ (1:m)] = dJdK*dKdB*dBdU
dJ

dJ - dJ_ana

dJN = zeros(n̄)
dJN[idx.e] = vec(Qfr')


f(ẋ,z) = prob.model.info[:fc](ẋ,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r))
∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
∇f(x,u) = ∇f([x;u])

function Bc(x,u)
    ∇f(x,u)[:,n .+ (1:m)]
end

Bc(z) = Bc(z[1:n],z[n .+ (1:m)])
∇Bc(z) = ForwardDiff.jacobian(Bc,z)


function ∇stage_cost(y)
    dJ = zeros(eltype(y),n̄+m)
    z = y[idx.z]
    u = y[length(idx.z) .+ (1:m)]
    _E = reshape(z[idx.e],n,n)
    _S = reshape(z[idx.s],n,n)
    _P = _S*_S'
    _K = K(z,u)
    _B = Bc(z)
    _∇B = ∇Bc([z[idx.x];u])

    dJdK = reshape(Rr*_K*_E + Rr'*_K*_E',1,m*n)
    dJdE = vec((Qr + _K'*Rr*_K)')
    dKdB = kron(_P',inv(Rr))*comm(n,m)
    dKdP = kron(Diagonal(ones(n)),(Rr)\_Bc[k]')
    dKdS = kron(_S,(Rr)\_B') + kron(Diagonal(ones(n)), (Rr)\(_B'*_S))*comm(n,n)
    dBdX = _∇B[:,1:n]
    dBdU = _∇B[:,n .+ (1:m)]

    dJ[idx.x] = dJdK*dKdB*dBdX
    dJ[idx.e] = dJdE
    dJ[idx.s] = dJdK*dKdS
    dJ[n̄ .+ (1:m)] = dJdK*dKdB*dBdU
    dJ
end

function ∇stage_cost_term(zN)
    dJN = zeros(eltype(zN),n̄)
    dJN[idx.e] = vec(Qfr')
    dJN
end

dJ_ana = ∇stage_cost([_Z[2];prob.U[2]])
dJN_ana = ∇stage_cost_term(_Z[2])

dJ - dJ_ana
dJN - dJN_ana
