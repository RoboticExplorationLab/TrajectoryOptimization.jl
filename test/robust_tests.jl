using Plots
p_model = Dynamics.pendulum_frictionless_model
n = p_model.n; m = p_model.m
x0 = [0.; 0.]
xf = [pi; 0.]

Q = 1.0*Diagonal(ones(n))
R = 1.0*Diagonal(ones(m))
Qf = 100.0*Diagonal(ones(n))

N = 201
tf = 1.0
dt = tf/(N-1)
U0 = [rand(m) for k = 1:N-1]
prob = TrajectoryOptimization.Problem(p_model, TrajectoryOptimization.Objective(LQRCost(Q,R,Qf,xf),N), integration=:rk3_implicit, x0=x0, N=N, tf=tf)
TrajectoryOptimization.initial_controls!(prob, U0)
# rollout!(prob)
solve!(prob,iLQRSolverOptions())
plot(prob.X)

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

"Time-varying Linear Quadratic Regulator; returns optimal linear feedback matrices and optimal cost-to-go"
function tvlqr_dis(prob::Problem{T},Q::AbstractArray{T},R::AbstractArray{T},Qf::AbstractArray{T}) where T
    n = prob.model.n; m = prob.model.m; N = prob.N
    dt = prob.dt

    K  = [zeros(T,m,n) for k = 1:N-1]
    ∇F = [PartedMatrix(zeros(T,n,length(prob.model)),create_partition2(prob.model)) for k = 1:N-1]
    P  = [zeros(T,n,n) for k = 1:N]

    # jacobian!(∇F,prob.model,prob.X,prob.U,prob.dt)

    P[N] .= Qf

    for k = N-1:-1:1
        jacobian!(∇F[k],prob.model,prob.X[k],prob.U[k],prob.dt)

        A, B = ∇F[k].xx, ∇F[k].xu
        K[k] .= (R*dt + B'*P[k+1]*B)\(B'*P[k+1]*A)
        P[k] .= Q*dt + K[k]'*R*K[k]*dt + (A - B*K[k])'*P[k+1]*(A - B*K[k])
        P[k] .= 0.5*(P[k] + P[k]')
    end

    return K, P
end

Kd,Pd = tvlqr_dis(prob,Q,R,Qf)

Pd_vec = [vec(Pd[k]) for k = 1:N]
plot(Pd_vec,legend=:left)

# P = [zeros(n^2) for k = 1:N]
# P[N] = vec(Qf)
#
# _t = [tf]
#
# for k = N:-1:2
#     k1 = k2 = k3 = k4 = zero(P[k])
#     x = P[k]
#     # println(_t[1] - dt)
#     r_dyn(k1, x, _t[1]);
#     k1 *= -dt;
#     r_dyn(k2, x + k1/2, _t[1] - dt/2);
#     k2 *= -dt;
#     r_dyn(k3, x + k2/2, _t[1] - dt/2);
#     k3 *= -dt;
#     r_dyn(k4, x + k3, max(_t[1] - dt, 0.));
#     k4 *= -dt;
#     copyto!(P[k-1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
#     _t[1] -= dt
#     # copyto!(P[k-1], x + k1)
# end
#
# Pv = [vec(P[k]) for k = 1:N]
# Pv[end-1]
# plot(Pv)
# # plot!(Pd_vec,legend=:top,linetype=:stairs)
#
# S = [zeros(n^2) for k = 1:N]
# S[N] = vec(cholesky(Qf).U)
# _t = [tf]
#
# for k = N:-1:2
#     k1 = k2 = k3 = k4 = zero(S[k])
#     x = S[k]
#     # println(_t[1] - dt)
#     r_dyn_sqrt(k1, x, _t[1]);
#     k1 *= -dt;
#     r_dyn_sqrt(k2, x + k1/2, _t[1] - dt/2);
#     k2 *= -dt;
#     r_dyn_sqrt(k3, x + k2/2, _t[1] - dt/2);
#     k3 *= -dt;
#     r_dyn_sqrt(k4, x + k3, max(_t[1] - dt, 0.));
#     k4 *= -dt;
#     copyto!(S[k-1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
#     _t[1] -= dt
#     # copyto!(P[k-1], x + k1)
# end
#
# Sv = [vec(S[k]) for k = 1:N]
# _Sv = [vec(reshape(S[k],n,n)*reshape(S[k],n,n)') for k = 1:N]
# plot(Sv)
# plot(_Sv)
# # plot!(Pd_vec,legend=:top,linetype=:stairs)
#
#
# P = [zeros(n^2) for k = 1:N]
# P[1] = Pv[1]
# P[1] = vec(reshape(S[1],n,n)*reshape(S[1],n,n)')
# # P[1] = vec(0.5*(reshape(Pd_vec[1],n,n) + reshape(Pd_vec[1],n,n)'))
# _t = [0.]
# for k = 1:N-1
#     k1 = k2 = k3 = k4 = zero(P[k])
#     x = P[k]
#     r_dyn(k1, x, _t[1]);
#     k1 *= dt;
#     r_dyn(k2, x + k1/2, _t[1] + dt/2);
#     k2 *= dt;
#     r_dyn(k3, x + k2/2, _t[1] + dt/2);
#     k3 *= dt;
#     r_dyn(k4, x + k3, min(_t[1] + dt,tf));
#     k4 *= dt;
#     copyto!(P[k+1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
#     _t[1] += dt
# end
# P[end]
#
# S = [zeros(n^2) for k = 1:N]
# S[1] = Sv[1]
#
# _t = [0.]
# for k = 1:N-1
#     k1 = k2 = k3 = k4 = zero(S[k])
#     x = S[k]
#     r_dyn_sqrt(k1, x, _t[1]);
#     k1 *= dt;
#     r_dyn_sqrt(k2, x + k1/2, _t[1] + dt/2);
#     k2 *= dt;
#     r_dyn_sqrt(k3, x + k2/2, _t[1] + dt/2);
#     k3 *= dt;
#     r_dyn_sqrt(k4, x + k3, min(_t[1] + dt,tf));
#     k4 *= dt;
#     copyto!(S[k+1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
#     _t[1] += dt
# end
# S[end]

## Implicit Midpoint
model = p_model
X = [zeros(n) for k = 1:N]
X[1] = copy(x0)

for k = 1:N-1
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

        Zc = zeros(n,n+m)
        model.∇f(Zc,Xm,prob.U[k])
        A = Zc[:,1:n]

        ∇g = Diagonal(I,n) - 0.5*dt*A
        δx = -∇g\g

        X[k+1] += δx
    end
end
Y = copy(X)
plot(Y)

X = [zeros(n) for k = 1:N]
X[N] = copy(Y[end])

for k = N-1:-1:1
    # get estimate of X[k+1] from explicit midpoint
    k1 = k2 = kg = zero(X[k+1])
    model.f(k1, X[k+1], prob.U[k]);
    k1 *= -dt;
    model.f(k2, X[k+1] + k1/2, prob.U[k]);
    k2 *= -dt;
    copyto!(X[k], X[k+1] + k2)

    # iterate to solve implicit midpoint step
    for i = 1:10
        Xm = 0.5*(X[k] + X[k+1])
        model.f(kg,Xm,prob.U[k])
        g = X[k] - X[k+1] + dt*kg

        Zc = zeros(n,n+m)
        model.∇f(Zc,Xm,prob.U[k])
        A = Zc[:,1:n]

        ∇g = Diagonal(I,n) + 0.5*dt*A
        δx = -∇g\g

        X[k] += δx
    end
end

plot(X)
plot!(Y)

X = [zeros(n) for k = 1:N]
X[1] = copy(x0)
for k = 1:N-1
    # get estimate of X[k+1] from explicit midpoint
    k1 = k2 = k3 = kg1 = kg2 = kg3 = zero(X[k])
    model.f(k1, X[k], prob.U[k]);
    k1 *= dt;
    model.f(k2, X[k] + k1/2, prob.U[k]);
    k2 *= dt;
    model.f(k3, X[k] - k1 + 2*k2, prob.U[k]);
    k3 *= dt;
    copyto!(X[k+1], X[k] + (k1 + 4*k2 + k3)/6)

    # iterate to solve implicit midpoint step
    for i = 1:10
        model.f(kg1,X[k],prob.U[k])
        model.f(kg3,X[k+1],prob.U[k])

        Xm = 0.5*(X[k] + X[k+1]) + dt/8*(kg1 - kg3)
        model.f(kg2,Xm,prob.U[k])

        g = X[k+1] - X[k] - dt/6*kg1 - 4/6*dt*kg2 - dt/6*kg3

        Zc1 = zeros(n,n+m)
        Zc2 = zeros(n,n+m)

        model.∇f(Zc1,Xm,prob.U[k])
        model.∇f(Zc2,X[k+1],prob.U[k])
        A1 = Zc1[:,1:n]
        A2 = Zc2[:,1:n]


        ∇g = Diagonal(I,n) - 4/6*dt*A1*(0.5*Diagonal(I,n) - dt/8*A2) - dt/6*A2
        δx = -∇g\g

        println(δx)
        X[k+1] += δx
    end
    println("\n")
end
Y = copy(X)
plot(Y)
plot!(prob.X)
Y[end]

X = [zeros(n) for k = 1:N]
# X[N] = copy(Y[end])
X[N] = copy(prob.X[end])

for k = N-1:-1:1
    # get estimate of X[k+1] from explicit midpoint
    k1 = k2 = k3 = kg1 = kg2 = kg3 = zero(X[k+1])
    model.f(k1, X[k+1], prob.U[k]);
    k1 *= -dt;
    model.f(k2, X[k+1] + k1/2, prob.U[k]);
    k2 *= -dt;
    model.f(k3, X[k+1] - k1 + 2*k2, prob.U[k]);
    k3 *= -dt;
    copyto!(X[k], X[k+1] + (k1 + 4*k2 + k3)/6)

    # iterate to solve implicit midpoint step
    # iterate to solve implicit midpoint step
    for i = 1:10
        model.f(kg1,X[k+1],prob.U[k])
        model.f(kg3,X[k],prob.U[k])

        Xm = 0.5*(X[k+1] + X[k]) - dt/8*(kg1 - kg3)
        model.f(kg2,Xm,prob.U[k])

        g = X[k] - X[k+1] + dt/6*kg1 + 4/6*dt*kg2 + dt/6*kg3

        Zc1 = zeros(n,n+m)
        Zc2 = zeros(n,n+m)

        model.∇f(Zc1,Xm,prob.U[k])
        model.∇f(Zc2,X[k],prob.U[k])
        A1 = Zc1[:,1:n]
        A2 = Zc2[:,1:n]


        ∇g = Diagonal(I,n) + 4/6*dt*A1*(0.5*Diagonal(I,n) + dt/8*A2) + dt/6*A2
        δx = -∇g\g

        println(δx)
        X[k] += δx
    end
    println("\n")
end

plot(Y)
plot!(X)
plot!(prob.X)

r_dyn_sqrt_wrap(ṡ,z) = r_dyn_sqrt(ṡ,z[1:n^2],z[n^2 + 1])
r_dyn_sqrt_wrap(rand(n^2),[rand(n^2);1.0])

∇r_dyn_sqrt_wrap(z) = ForwardDiff.jacobian(r_dyn_sqrt_wrap,zeros(n^2),z)

∇r_dyn_sqrt_wrap(s,t) = ∇r_dyn_sqrt_wrap([s;t])

∇r_dyn_sqrt_wrap([rand(n^2);1.0])

S = [zeros(n^2) for k = 1:N]
S[N] = vec(cholesky(Qf).U)
_t = [tf]

for k = N:-1:2
    # println(k)
    # explicit midpoint
    k1 = k2 = kg = zero(S[k])
    x = S[k]
    r_dyn_sqrt(k1, S[k], _t[1]);
    k1 *= -dt;
    r_dyn_sqrt(k2, S[k] + k1/2, _t[1] - dt/2);
    k2 *= -dt;
    copyto!(S[k-1], S[k] + k2)

    for i = 1:25
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
plot(S)
Ps = [vec(reshape(S[k],n,n)*reshape(S[k],n,n)') for k = 1:N]
plot(Ps)
plot!(Pd_vec,linetype=:steppost)
Ps[1]
S1 = copy(S[1])

S = [zeros(n^2) for k = 1:N]
S[1] = S1
# S[1] = vec(sqrt(Pd[1]))
_t = [0.]

for k = 1:N-1
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

plot!(S,legend=:left)
Sp_f = [vec(reshape(S[k],n,n)*reshape(S[k],n,n)') for k = 1:N]
plot!(Sp_f)

## Implicit with full robust dynamics
fr = Dynamics.pendulum_dynamics_uncertain!
n = 2; m = 1; r = 1

_fr(ẋ,z) = fr(ẋ,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
∇fr(Z,z) = ForwardDiff.jacobian!(Z,_fr,zeros(eltype(z),n),z)
∇fr(Z,x,u,w) = ∇fr(Z,[x;u;w])
E1 = 1.0e-6*ones(n,n)
H1 = zeros(n,r)
D = [0.2^2]

idx = (x=1:n,u=1:m,w=1:r)
z_idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)))

Zc = zeros(n,n+m+r)
n̄ = n + n^2 + n*r + n^2

function dp_dyn(ż,z,u,w)
    x = z[z_idx.x]
    E = reshape(z[z_idx.e],n,n)
    H = reshape(z[z_idx.h],n,r)
    _S = reshape(z[z_idx.s],n,n)
    ss = zeros(eltype(z),n^2)
    try
        ss = inv(_S')
    catch
        println(_S')
        error("no inv S")
    end

    _P = _S*_S'

    Zc = zeros(eltype(z),n,n+m+r)
    ∇fr(Zc,x,u,w)

    Ac = Zc[:,idx.x]
    Bc = Zc[:,n .+ idx.u]
    Gc = Zc[:,(n+m) .+ idx.w]

    Kc = R\(Bc'*_P)
    Acl = Ac - Bc*Kc

    fr(view(ż,z_idx.x),x,u,w)
    ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
    ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
    ż[z_idx.s] = reshape(-.5*Q*ss - Ac'*_S + .5*(_P*Bc)*(R\(Bc'*_S)),n^2)
end

function dp_dyn_nom(ż,z,u,w)
    x = z[z_idx.x]
    E = reshape(z[z_idx.e],n,n)
    H = reshape(z[z_idx.h],n,r)
    _S = reshape(z[z_idx.s],n,n)
    ss = inv(_S')
    _P = _S*_S'

    Zc = zeros(eltype(z),n,n+m+r)
    ∇fr(Zc,x,u,w)

    Ac = Zc[:,idx.x]
    Bc = Zc[:,n .+ idx.u]
    Gc = Zc[:,(n+m) .+ idx.w]

    Kc = R\(Bc'*_P)
    Acl = Ac - Bc*Kc

    fr(view(ż,z_idx.x),x,u,w)
    # ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
    # ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
    ż[z_idx.s] = reshape(-.5*Q*ss - Ac'*_S + .5*(_P*Bc)*(R\(Bc'*_S)),n^2)
end

dp_dyn(ż,zz) = dp_dyn(ż,zz[1:n̄],zz[n̄ .+ (1:m)],zz[(n̄+m) .+ (1:r)])
∇dp_dyn(Z,zz) = ForwardDiff.jacobian(dp_dyn,zeros(n̄),zz)
∇dp_dyn(Z,z,u,w) = ∇dp_dyn(Z,[z;u;w])

Z = [zeros(n̄) for k = 1:N]
Z[1][z_idx.x] = x0
Z[1][z_idx.e] = E1
Z[1][z_idx.h] = H1
Z[1][z_idx.s] = vec(sqrt(Pd[1]))

for k = 1:N-1
    # get estimate of X[k+1] from explicit midpoint
    k1 = k2 = kg = zero(Z[k])
    dp_dyn_nom(k1, Z[k], prob.U[k], zeros(r));
    k1 *= dt;
    dp_dyn_nom(k2, Z[k] + k1/2, prob.U[k], zeros(r));
    k2 *= dt;
    copyto!(Z[k+1], Z[k] + k2)

    # iterate to solve implicit midpoint step
    for i = 1:10
        Zm = 0.5*(Z[k] + Z[k+1])
        dp_dyn_nom(kg,Zm,prob.U[k],zeros(r))
        g = Z[k+1] - Z[k] - dt*kg

        ZZ = zeros(n̄,n̄+m+r)
        ∇dp_dyn(ZZ,Zm,prob.U[k],zeros(r))
        _A = ZZ[:,1:n̄]

        ∇g = Diagonal(I,n̄) - 0.5*dt*_A
        δz = -∇g\g

        Z[k+1] += δz
    end
end

Z = [zeros(n̄) for k = 1:N]
Z[N][z_idx.x] = xf
Z[N][z_idx.s] = vec(sqrt(Qf))

for k = N:-1:2
    println(k)
    # get estimate of X[k+1] from explicit midpoint
    k1 = k2 = kg = zero(Z[k])
    dp_dyn_nom(k1, Z[k], prob.U[k-1], zeros(r));
    k1 *= -dt;
    dp_dyn_nom(k2, Z[k] + k1/2, prob.U[k-1], zeros(r));
    k2 *= -dt;
    copyto!(Z[k-1], Z[k] + k2)

    # iterate to solve implicit midpoint step
    for i = 1:10
        Zm = 0.5*(Z[k] + Z[k-1])
        dp_dyn_nom(kg,Zm,prob.U[k-1],zeros(r))
        g = Z[k-1] - Z[k] + dt*kg

        ZZ = zeros(n̄,n̄+m+r)
        ∇dp_dyn(ZZ,Zm,prob.U[k-1],zeros(r))
        _A = ZZ[:,1:n̄]

        ∇g = Diagonal(I,n̄) + 0.5*dt*_A
        δz = -∇g\g

        Z[k-1] += δz
    end
end


Z = [zeros(n̄) for k = 1:N]
Z[1][z_idx.x] = x0
Z[1][z_idx.e] = E1
Z[1][z_idx.h] = H1
Z[1][z_idx.s] = vec(sqrt(Pd[1]))


for k = 1:N-1
    # get estimate of X[k+1] from explicit midpoint
    k1 = k2 = kg = zero(Z[k])
    dp_dyn(k1, Z[k], prob.U[k], zeros(r));
    k1 *= dt;
    dp_dyn(k2, Z[k] + k1/2, prob.U[k], zeros(r));
    k2 *= dt;
    copyto!(Z[k+1], Z[k] + k2)
    println(k)
    # iterate to solve implicit midpoint step
    for i = 1:10
        Zm = 0.5*(Z[k] + Z[k+1])
        dp_dyn(kg,Zm,prob.U[k],zeros(r))
        g = Z[k+1] - Z[k] - dt*kg

        ZZ = zeros(n̄,n̄+m+r)
        ∇dp_dyn(ZZ,Zm,prob.U[k],zeros(r))
        _A = ZZ[:,1:n̄]

        ∇g = Diagonal(I,n̄) - 0.5*dt*_A
        δz = -∇g\g

        Z[k+1] += δz
    end
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
