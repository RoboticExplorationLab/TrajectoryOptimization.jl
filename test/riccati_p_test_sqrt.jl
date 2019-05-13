using Plots
p_model = Dynamics.pendulum_model
n = p_model.n; m = p_model.m
x0 = [0.; 0.]
xf = [pi; 0.]

Q = 1.0*Diagonal(ones(n))
R = 1.0*Diagonal(ones(m))
Qf = 10.0*Diagonal(ones(n))
cholesky(Qf).U

N = 101
tf = 1.0
dt = tf/(N-1)
U0 = [rand(m) for k = 1:N-1]

prob = TrajectoryOptimization.Problem(p_model, TrajectoryOptimization.Objective(LQRCost(Q,R,Qf,xf),N), integration=:rk3, x0=x0, N=N, tf=tf)
TrajectoryOptimization.initial_controls!(prob, U0)
# rollout!(prob)
solve!(prob,iLQRSolverOptions())
Kd, Pd = tvlqr_dis(prob,Q,R,Qf)

Pd_vec = [vec(Pd[k]) for k = 1:N]
plot(Pd_vec,linetype=:steppost,legend=:top)

X_interp = gen_cubic_interp(prob.X,prob.dt)
U_interp = gen_zoh_interp([prob.U...,prob.U[end]],prob.dt)

function _Ac(t)
    Zc = zeros(n,n+m)
    p_model.∇f(Zc,zeros(n),X_interp(t),U_interp(t))
    Zc[:,1:n]
end

function _Bc(t)
    Zc = zeros(n,n+m)
    p_model.∇f(Zc,zeros(n),X_interp(t),U_interp(t))
    Zc[:,n .+ (1:m)]
end

function r_dyn(ṗ,p,t)
    P = reshape(p,n,n)
    P = 0.5*(P + P')

    # ee = eigen(P)
    # for i = 1:length(ee.values)
    #     if ee.values[i] <= 0.
    #         ee.values[i] = 1e-6
    #     end
    # end
    # P = ee.vectors*Diagonal(ee.values)*ee.vectors'
    # if !isposdef(P)
    #     error("not pos def")
    # end
    ṗ[1:n^2] = reshape(-1.0*(_Ac(t)'*P + P*_Ac(t) - P*_Bc(t)*(R\*(_Bc(t)'*P)) + Q),n^2)
end

function r_dyn_sqrt(ṡ,s,t)
    S = reshape(s,n,n)

    # ee = eigen(P)
    # for i = 1:length(ee.values)
    #     if ee.values[i] <= 0.
    #         ee.values[i] = 1e-6
    #     end
    # end
    # P = ee.vectors*Diagonal(ee.values)*ee.vectors'
    # if !isposdef(P)
    #     error("not pos def")
    # end
    ss = inv(S')
    ṡ[1:n^2] = -.5*Q*ss - _Ac(t)'*S + .5*(S*S'*_Bc(t))*inv(R)*(_Bc(t)'*S);
    # Sorig = S*S';
end

NN = floor(Int64,N)
dt = tf/(NN-1)
P = [zeros(n^2) for k = 1:NN]
P[NN] = vec(Qf)

S = [zeros(n^2) for k = 1:NN]
S[NN] = vec(cholesky(Qf).U)


_t = [tf]

for k = NN:-1:2
    k1 = k2 = k3 = k4 = zero(P[k])
    x = P[k]
    # println(_t[1] - dt)
    r_dyn(k1, x, _t[1]);
    k1 *= -dt;
    r_dyn(k2, x + k1/2, _t[1] - dt/2);
    k2 *= -dt;
    r_dyn(k3, x + k2/2, _t[1] - dt/2);
    k3 *= -dt;
    r_dyn(k4, x + k3, max(_t[1] - dt, 0.));
    k4 *= -dt;
    copyto!(P[k-1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
    _t[1] -= dt
    # copyto!(P[k-1], x + k1)
end

Pv = [vec(P[k]) for k = 1:NN]

plot(Pv)
plot!(Pd_vec,legend=:top,linetype=:stairs)

_t = [tf]

for k = NN:-1:2
    k1 = k2 = k3 = k4 = zero(S[k])
    x = S[k]
    # println(_t[1] - dt)
    r_dyn_sqrt(k1, x, _t[1]);
    k1 *= -dt;
    r_dyn_sqrt(k2, x + k1/2, _t[1] - dt/2);
    k2 *= -dt;
    r_dyn_sqrt(k3, x + k2/2, _t[1] - dt/2);
    k3 *= -dt;
    r_dyn_sqrt(k4, x + k3, max(_t[1] - dt, 0.));
    k4 *= -dt;
    copyto!(S[k-1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
    _t[1] -= dt
    # copyto!(P[k-1], x + k1)
end

Sv = [vec(S[k]) for k = 1:NN]
_Sv = [vec(reshape(S[k],n,n)*reshape(S[k],n,n)') for k = 1:NN]
plot(Sv)
plot(_Sv)
plot!(Pd_vec,legend=:top,linetype=:stairs)




P = [zeros(n^2) for k = 1:NN]
P[1] = Pv[1]

_t = [0.]
for k = 1:NN-1
    k1 = k2 = k3 = k4 = zero(P[k])
    x = P[k]
    r_dyn(k1, x, _t[1]);
    k1 *= dt;
    r_dyn(k2, x + k1/2, _t[1] + dt/2);
    k2 *= dt;
    r_dyn(k3, x + k2/2, _t[1] + dt/2);
    k3 *= dt;
    r_dyn(k4, x + k3, min(_t[1] + dt,tf));
    k4 *= dt;
    copyto!(P[k+1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
    _t[1] += dt
end

plot(P)
plot!(Pd_vec,legend=:top)


idx = (x=1:n,u=1:m)
z_idx = (x=1:n,p=(n .+ (1:n^2)))

Zc = zeros(n,n+m)
n̄ = n + n^2

function dp_dyn(ż,z,u)
    x = z[z_idx.x]
    P = reshape(z[z_idx.p],n,n)
    P = 0.5*(P + P')

    p_model.∇f(Zc,x,u)
    Ac = Zc[:,idx.x]
    Bc = Zc[:,n .+ idx.u]

    Kc = R\(Bc'*P)

    p_model.f(view(ż,z_idx.x),x,u)
    ż[z_idx.p] = reshape(-1.0*(Ac'*P + P*Ac - P*Bc*Kc + Q),n^2)
end

Z = [zeros(n̄) for k = 1:NN]
Z[1][z_idx.x] = x0
Z[1][z_idx.p] = vec(P[1])

for k = 1:NN-1
    k1 = k2 = k3 = k4 = zeros(n̄)
    dp_dyn(k1, Z[k], prob.U[k]);
    k1 *= dt;
    dp_dyn(k2, Z[k] + k1/2, prob.U[k]);
    k2 *= dt;
    dp_dyn(k3, Z[k] + k2/2, prob.U[k]);
    k3 *= dt;
    dp_dyn(k4, Z[k] + k3, prob.U[k]);
    k4 *= dt;
    copyto!(Z[k+1], Z[k] + (k1 + 2*k2 + 2*k3 + k4)/6)
end

plot(P)
