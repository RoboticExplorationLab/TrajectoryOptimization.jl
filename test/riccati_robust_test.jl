using Plots
p_model = Dynamics.pendulum_model_uncertain
n = p_model.n; m = p_model.m; r = p_model.r
x0 = [0.; 0.]
xf = [pi; 0.]

E1 = 1.0e-6*ones(n,n)
H1 = zeros(n,r)
D = [0.2^2]

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
    Zc = zeros(n,n+m+r)
    p_model.∇f(Zc,zeros(n),X_interp(t),U_interp(t),zeros(r))
    Zc[:,1:n]
end

function _Bc(t)
    Zc = zeros(n,n+m+r)
    p_model.∇f(Zc,zeros(n),X_interp(t),U_interp(t),zeros(r))
    Zc[:,n .+ (1:m)]
end

function _Gc(t)
    Zc = zeros(n,n+m+r)
    p_model.∇f(Zc,zeros(n),X_interp(t),U_interp(t),zeros(r))
    Zc[:,(n + m) .+ (1:r)]
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


idx = (x=1:n,u=1:m,w=1:r)
z_idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)))

Zc = zeros(n,n+m+r)
n̄ = n + n^2 + n*r + n^2

function dp_dyn(ż,z,u,w)
    x = z[z_idx.x]
    E = reshape(z[z_idx.e],n,n)
    H = reshape(z[z_idx.h],n,r)
    S = reshape(z[z_idx.s],n,n)
    ss = inv(S')
    P = S*S'

    p_model.∇f(Zc,x,u,w)
    Ac = Zc[:,idx.x]
    Bc = Zc[:,n .+ idx.u]
    Gc = Zc[:,(n+m) .+ idx.w]

    Kc = R\(Bc'*P)
    Acl = Ac - Bc*Kc

    p_model.f(view(ż,z_idx.x),x,u,w)
    ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
    ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
    ż[z_idx.s] = reshape(-.5*Q*ss - Ac'*S + .5*(P*Bc)*(R\(Bc'*S)),n^2)
end

Z = [zeros(n̄) for k = 1:NN]
Z[1][z_idx.x] = x0
Z[1][z_idx.e] = E1
Z[1][z_idx.h] = H1
Z[1][z_idx.s] = vec(sqrt(reshape(P[1],n,n)))

for k = 1:NN-1
    k1 = k2 = k3 = k4 = zeros(n̄)
    dp_dyn(k1, Z[k], prob.U[k], zeros(r));
    k1 *= dt;
    dp_dyn(k2, Z[k] + k1/2, prob.U[k], zeros(r));
    k2 *= dt;
    dp_dyn(k3, Z[k] + k2/2, prob.U[k], zeros(r));
    k3 *= dt;
    dp_dyn(k4, Z[k] + k3, prob.U[k], zeros(r));
    k4 *= dt;
    copyto!(Z[k+1], Z[k] + (k1 + 2*k2 + 2*k3 + k4)/6)
end


xx = [Z[k][z_idx.x] for k = 1:N]
ee = [Z[k][z_idx.e] for k = 1:N]
hh = [Z[k][z_idx.h] for k = 1:N]
ss = [Z[k][z_idx.s] for k = 1:N]
pp = [vec(reshape(Z[k][z_idx.s],n,n)*reshape(Z[k][z_idx.s],n,n)') for k = 1:N]


plot(xx)
plot(ee)
plot(hh)
plot(ss)
plot(pp)

prob_robust = robust_problem(prob,E1,H1,D,Q,R,Qf,Q,R,Qf)
