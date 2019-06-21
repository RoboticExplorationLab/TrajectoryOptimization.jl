using Plots, ForwardDiff, OrdinaryDiffEq, ODEInterface, ODEInterfaceDiffEq
using ODEInterfaceDiffEq
ODEInterfaceDiffEq.radau5

model = Dynamics.cartpole_model
n = model.n; m = model.m

x0 = [0.; 0.; 0.; 0.]
xf = [pi; 0.; 0.; 0.]

Q = 1.0e-2*Diagonal(ones(n))
R = 1.0e-2*Diagonal(ones(m))
Qf = 1000.0*Diagonal(ones(n))

# Qr = 1.0*Diagonal(I,n)
# Qfr = 100.0*Diagonal(I,n)
# Rr = 1.0*Diagonal(I,m)

Qr = Diagonal([10.;0.])
Qfr = 100.0*Diagonal(I,n)
Rr = 1.0e-1*Diagonal(I,m)

f(ẋ,z) = model.f(ẋ,z[1:n],z[n .+ (1:m)])
∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
∇f(x,u) = ∇f([x;u])

function robust_dynamics(ż,z,u)
    w = zeros(r)
    x = z[1:n]
    S = reshape(z[n .+ (1:n^2)],n,n)
    ss = inv(S')
    P = S*S'

    Zc = ∇f([x;u])
    Ac = Zc[:,1:n]
    Bc = Zc[:,n .+ (1:m)]

    Kc = Rr\(Bc'*P)
    Acl = Ac - Bc*Kc

    f(view(ż,1:n),[x;u])
    ż[n .+ (1:n^2)] = reshape(-.5*Qr*ss - Ac'*S + .5*P*Bc*(Rr\(Bc'*S)),n^2)
end





costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,constraint_tolerance=1.0e-3)

N = 201
tf = 2.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:rk3_implicit,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)

initial_controls!(prob,U0)
solve!(prob,opts_ilqr)

plot(prob.X)
plot(prob.U)

Kd,Pd = tvlqr_dis(prob,Qr,Rr,Qfr)
Kc,Sc = tvlqr_sqrt_con_rk3(prob,Qr,Rr,Qfr,xf)

Pd_vec = [vec(Pd[k]) for k = 1:N]
Pc_vec = [vec(reshape(Sc[k],n,n)*reshape(Sc[k],n,n)') for k = 1:N]
S1 = copy(Sc[1])

# @test norm(Pd_vec[1] - Pc_vec[1]) < 0.1
plot(Pd_vec,label="",linetype=:steppost)
plot!(Pc_vec,label="")

Z = [zeros(n+n^2) for k = 1:N]
Z[1] = [x0;S1]

for k = 1:N-1
    _u = prob.U[k]

    function dyn(p,w,t)
        ṗ = zero(p)
        robust_dynamics(ṗ,p,_u)
        return ṗ
    end

    _tf = dt*k
    _t0 = dt*(k-1)

    u0=vec(Z[k])
    tspan = (_t0,_tf)
    pro = ODEProblem(dyn,u0,tspan)
    sol = OrdinaryDiffEq.solve(pro,RadauIIA5(),dt=dt)#,reltol=1e-8,abstol=1e-8)
    Z[k+1] = sol.u[end]
end

plot(prob.X)

zz = [Z[k][1:n] for k = 1:N]

plot!(zz)

ss = [Z[k][n .+ (1:n^2)] for k = 1:N]
pp = [vec(reshape(Z[k][n .+ (1:n^2)],n,n)*reshape(Z[k][n .+ (1:n^2)],n,n)') for k = 1:N]
plot(ss,label="")
plot(pp)
