using Plots
p_model = Dynamics.pendulum_model
n = p_model.n; m = p_model.m
x0 = [0.; 0.]
xf = [pi; 0.]

Q = 1.0*Diagonal(ones(n))
R = 1.0*Diagonal(ones(m))
Qf = 10.0*Diagonal(ones(n))

N = 101
tf = 1.0
dt = tf/(N-1)
U0 = [rand(m) for k = 1:N-1]

prob = TrajectoryOptimization.Problem(p_model, TrajectoryOptimization.Objective(LQRCost(Q,R,Qf,xf),N), integration=:rk3, x0=x0, N=N, tf=tf)
TrajectoryOptimization.initial_controls!(prob, U0)
# rollout!(prob)
solve!(prob,iLQRSolverOptions())


function gen_cubic_interp(X,dt)
    N = length(X); n = length(X[1])

    function interp(t)
        j = t/dt + 1.
        x = zeros(eltype(t),n)
        for i = 1:n
            x[i] = interpolate([X[k][i] for k = 1:N],BSpline(Cubic(Line(OnGrid()))))(j)
        end
        return x
    end
end

function gen_zoh_interp(X,dt)
    N = length(X); n = length(X[1])

    function interp(t)
        if (t/dt + 1.0)%floor(t/dt + 1.0) < 0.99999
            j = convert(Int64,floor(t/dt)) + 1
        else
            j = convert(Int64,ceil(t/dt)) + 1
        end

        interpolate(X,BSpline(Constant()))(j)

    end
end

X_interp = gen_cubic_interp(prob.X,prob.dt)
U_interp = gen_zoh_interp([prob.U...,prob.U[end]],prob.dt)


f(ẋ,z) = p_model.f(ẋ,z[1:n],z[n .+ (1:m)])
∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
∇f(x,u) = ∇f([x;u])

function _Ac(t)
    # Zc = zeros(eltype(t),n,n+m)
    # p_model.∇f(Zc,zeros(eltype(t),n),X_interp(t),U_interp(t))
    # Zc[:,1:n]
    ∇f(X_interp(t),U_interp(t))[:,1:n]
end


function _Bc(t)
    # Zc = zeros(eltype(t),n,n+m)
    # p_model.∇f(Zc,zeros(eltype(t),n),X_interp(t),U_interp(t))
    # Zc[:,n .+ (1:m)]
    ∇f(X_interp(t),U_interp(t))[:,n .+ (1:m)]
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
    ṡ[1:n^2] = vec(-.5*Q*ss - _Ac(t)'*S + .5*(S*S'*_Bc(t))*inv(R)*(_Bc(t)'*S));
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

# plot(Pv)
# plot!(Pd_vec,legend=:top,linetype=:stairs)

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
# plot(Sv)
# plot(_Sv)
# plot!(Pd_vec,legend=:top,linetype=:stairs)


P = [zeros(n^2) for k = 1:NN]
P[1] = Pv[1]
P[1] = vec(reshape(S[1],n,n)*reshape(S[1],n,n)')
# P[1] = vec(0.5*(reshape(Pd_vec[1],n,n) + reshape(Pd_vec[1],n,n)'))
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
P[end]

S = [zeros(n^2) for k = 1:NN]
S[1] = Sv[1]

_t = [0.]
for k = 1:NN-1
    k1 = k2 = k3 = k4 = zero(S[k])
    x = S[k]
    r_dyn_sqrt(k1, x, _t[1]);
    k1 *= dt;
    r_dyn_sqrt(k2, x + k1/2, _t[1] + dt/2);
    k2 *= dt;
    r_dyn_sqrt(k3, x + k2/2, _t[1] + dt/2);
    k3 *= dt;
    r_dyn_sqrt(k4, x + k3, min(_t[1] + dt,tf));
    k4 *= dt;
    copyto!(S[k+1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
    _t[1] += dt
end
S[end]

## Implicit Midpoint
model = p_model
Zc = zeros(n,n+m)
X = [zeros(n) for k = 1:NN]
X[1] = copy(x0)

for k = 1:NN-1
    # get estimate of X[k+1] from explicit midpoint
    k1 = k2 = kg = zero(X[k])
    model.f(k1, X[k], prob.U[k]);
    k1 *= dt;
    model.f(k2, X[k] + k1/2, prob.U[k]);
    k2 *= dt;
    copyto!(X[k+1], X[k] + k2)

    # iterate to solve implicit midpoint step
    for i = 1:10
        Xm = 0.5*(X[k] + X[k+1])
        model.f(kg,Xm,prob.U[k])
        g = X[k+1] - X[k] - dt*kg

        model.∇f(Zc,Xm,prob.U[k])
        A = Zc[:,1:n]

        ∇g = Diagonal(I,n) - 0.5*dt*A
        δx = -∇g\g

        X[k+1] += δx
    end
end

plot(prob.X)
plot!(X)

norm(X[end] - prob.X[end])

r_dyn_sqrt(rand(n^2),rand(n^2),1.0)
r_dyn_sqrt_wrap(ṡ,z) = r_dyn_sqrt(ṡ,z[1:n^2],z[n^2 + 1])
r_dyn_sqrt_wrap(rand(n^2),[rand(n^2);1.0])

∇r_dyn_sqrt_wrap(z) = ForwardDiff.jacobian(r_dyn_sqrt_wrap,zeros(n^2),z)

∇r_dyn_sqrt_wrap(s,t) = ∇r_dyn_sqrt_wrap([s;t])

∇r_dyn_sqrt_wrap([rand(n^2);1.0])

S = [zeros(n^2) for k = 1:NN]
S[NN] = vec(cholesky(Qf).U)
_t = [tf]

for k = NN:-1:2
    # explicit midpoint
    k1 = k2 = kg = zero(S[k])
    x = S[k]
    r_dyn_sqrt(k1, S[k], _t[1]);
    k1 *= -dt;
    r_dyn_sqrt(k2, S[k] + k1/2, _t[1] - dt/2);
    k2 *= -dt;
    copyto!(S[k-1], S[k] + k2)

    for i = 1:10
        Sm = 0.5*(S[k-1] + S[k])
        tm = _t[1] - dt/2

        r_dyn_sqrt(kg,Sm,tm)
        g = S[k-1] - S[k] + dt*kg

        A = ∇r_dyn_sqrt_wrap(Sm,tm)[:,1:n^2]

        ∇g = Diagonal(I,n^2) + 0.5*dt*A
        δs = -∇g\g

        S[k-1] += δs
    end

    _t[1] -= dt
end

S1 = copy(S[1])

S = [zeros(n^2) for k = 1:NN]
S[1] = S1
_t = [0.]

for k = 1:NN-1
    # explicit midpoint
    k1 = k2 = kg = zero(S[k])

    r_dyn_sqrt(k1, S[k], _t[1]);
    k1 *= dt;
    r_dyn_sqrt(k2, S[k] + k1/2, _t[1] + dt/2);
    k2 *= dt;
    copyto!(S[k+1], S[k] + k2)

    for i = 1:10
        Sm = 0.5*(S[k+1] + S[k])
        tm = _t[1] + dt/2

        r_dyn_sqrt(kg,Sm,tm)
        g = S[k+1] - S[k] - dt*kg

        A = ∇r_dyn_sqrt_wrap(Sm,tm)[:,1:n^2]

        ∇g = Diagonal(I,n^2) - 0.5*dt*A
        δs = -∇g\g

        S[k+1] += δs
    end

    _t[1] += dt
end

S[end]
