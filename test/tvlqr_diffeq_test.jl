using Plots, ForwardDiff, OrdinaryDiffEq, ODEInterface, ODEInterfaceDiffEq
using ODEInterfaceDiffEq

T = Float64
model = Dynamics.cartpole_model_uncertain
n = model.n; m = model.m; r = model.r

x0 = [0.; 0.; 0.; 0.]
xf = [0.; pi; 0.; 0.]

Q = 1.0e-2*Diagonal(ones(n))
R = 1.0e-2*Diagonal(ones(m))
Qf = 1000.0*Diagonal(ones(n))

Qr = 1.0*Diagonal(I,n)
Qfr = 100.0*Diagonal(I,n)
Rr = 1.0*Diagonal(I,m)

# Qr = Diagonal([10.;0.])
# Qfr = 100.0*Diagonal(I,n)
# Rr = 1.0e-1*Diagonal(I,m)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,constraint_tolerance=1.0e-3)

N = 201
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:DiffEq_ImplicitMidpoint,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)

initial_controls!(prob,U0)
solve!(prob,opts_ilqr)

plot(prob.X)
plot(prob.U)

# K,P,Ps = tvlqr_con(prob,Qr,Rr,Qfr,xf,:ImplicitMidpoint)
Kd,Pd = tvlqr_dis(prob,Qr,Rr,Qfr)
K,P,Ps,Ac,Bc,Gc = tvlqr_con_uncertain(prob,Qr,Rr,Qfr,xf,:ImplicitMidpoint)

_Pd = [vec(Pd[k]) for k = 1:N]
_Ps = [vec(Ps[k]) for k = 1:N]
_P = [vec(Ps[k]*Ps[k]') for k = 1:N]
plot(_Ps)
plot(_P[150:end],label="")
plot!(_Pd[150:end],label="")

_Pd[end]
_P[end]

Ps[end]*Ps[end]'

Qfr


D = Diagonal((0.2^2)*ones(r))
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

E = [zeros(n,n) for k = 1:N]
H = [zeros(n,r) for k = 1:N]

E[1] = E1
H[1] = H1

for k = 1:N-1
    Acl = Ac[k] - Bc[k]*K[k]
    E[k+1] = Acl*E[k]*Acl' + Gc[k]*H[k]'*Acl' + Acl*H[k]*Gc[k]' + Gc[k]*D*Gc[k]'
    H[k+1] = Acl*H[k] + Gc[k]*D
end
