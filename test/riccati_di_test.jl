using Plots
n = 2; m = 1

x0 = [0.; 0.]
xf = [1.; 0.]
Ac = [0. 1.; 0. 0.]; Bc = [0.; 1.]

Q = 1.0*Diagonal(ones(n))
R = 1.0*Diagonal(ones(m))
Qf = 100.0*Diagonal(ones(n))

function di_dyn(ẋ,x,u)
    ẋ[1:2] = Ac*x[1:2] + Bc*u[1]
end

function r_dyn(ṗ,p)
    P = reshape(p,n,n)
    P = 0.5*(P + P')

    ee = eigen(P)
    for i = 1:length(ee.values)
        if ee.values[i] <= 0.
            ee.values[i] = 1e-6
        end
    end
    P = ee.vectors*Diagonal(ee.values)*ee.vectors'
    # if !isposdef(P)
    #     error("not pos def")
    # end
    ṗ[1:n^2] = reshape(-1.0*(Ac'*P + P*Ac - P*Bc*(R\*(Bc'*P)) + Q),n^2)
end


di_model = Model(di_dyn,n,m)

Zc = zeros(n,n+m)
di_model.∇f(Zc,rand(n),rand(n),rand(m))
Zc


N = 300
tf = 5.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

prob = TrajectoryOptimization.Problem(di_model, TrajectoryOptimization.Objective(LQRCost(Q,R,Qf,xf),N), integration=:rk4, x0=x0, N=N, tf=tf)
TrajectoryOptimization.initial_controls!(prob, U0)
rollout!(prob)
Kd, Pd = tvlqr_dis(prob,Q,R,Qf)
Pd_vec = [vec(Pd[k]) for k = 1:N]
plot(Pd_vec,linetype=:steppost,legend=:top)

X = [zeros(n) for k = 1:N]

di_dis = rk3(di_dyn,dt)
X[1] = x0
for k = 1:N-1
    di_dis(X[k+1],X[k],U0[k],dt)
end
plot(X)
plot!(prob.X)

NN = floor(Int64,10*N)
dt = tf/(NN-1)
P = [zeros(n^2) for k = 1:NN]
P[NN] = vec(Qf)
# r_dis = rk3(r_dyn,dt)

for k = NN:-1:2
    k1 = k2 = k3 = k4 = zero(P[k])
    x = P[k]
    println(k)
    r_dyn(k1, x);
    k1 *= -dt;
    r_dyn(k2, x + k1/2);
    k2 *= -dt;
    r_dyn(k3, x + k2/2);
    k3 *= -dt;
    r_dyn(k4, x + k3);
    k4 *= -dt;
    copyto!(P[k-1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
    # copyto!(P[k-1], x + k1)
end

Pv = [vec(P[k]) for k = 1:NN]

plot(Pv)
plot!(Pd_vec,legend=:top)

Pv[1]
Pd_vec[1]

P = [zeros(n^2) for k = 1:NN]
P[1] = Pv[1]

for k = 1:NN-1
    k1 = k2 = k3 = k4 = zero(P[k])
    x = P[k]
    r_dyn(k1, x);
    k1 *= dt;
    r_dyn(k2, x + k1/2);
    k2 *= dt;
    r_dyn(k3, x + k2/2);
    k3 *= dt;
    r_dyn(k4, x + k3);
    k4 *= dt;
    copyto!(P[k+1], x + (k1 + 2*k2 + 2*k3 + k4)/6)
end

plot(P)
plot!(Pd_vec,legend=:top)
