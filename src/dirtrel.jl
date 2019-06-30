# Pendulum Swing-Up Direct Transcription
using TrajectoryOptimization, LinearAlgebra, Plots, ForwardDiff, MatrixCalculus, Ipopt, MathOptInterface, BenchmarkTools
const MOI = MathOptInterface

model = Dynamics.pendulum_model_uncertain
n = model.n; m = model.m; r = model.r

eig_thr = 1.0e-3
n = 2; m = 1; r = 1# states, controls

u_max = 3.
u_min = -3.

h_max = Inf
h_min = 0.0

# Problem
x0 = [0.;0.]
xf = [pi;0.]

E0 = Diagonal(1.0e-6*ones(n))
H0 = zeros(n,r)
D = Diagonal([.2^2])

N = 51 # knot points
NN = (n+m+1)*(N-1) + n # number of decision variables

p_dynamics = n*(N+1) + (N-2)# number of equality constraints: dynamics, xf, h_k = h_{k+1}

# p += 2*n^2 # robust terminal constraint n^2
# p_ineq = 2*(m^2)*(N-1)*2

Q = Diagonal(zeros(n))
R = Diagonal(zeros(m))
Qf = Diagonal(zeros(n))

Q_lqr = Diagonal([10.;1.])
R_lqr = Diagonal(0.1*ones(m))
Qf_lqr = Diagonal([100.;100.])

Q_r = Q_lqr
R_r = R_lqr
Qf_r = Qf_lqr

tf0 = 2.
dt = tf0/(N-1)

function line_trajectory(x0::Vector,xf::Vector,N::Int)
    t = range(0,stop=N,length=N)
    slope = (xf .- x0)./N
    x_traj = [slope*t[k] for k = 1:N]
    x_traj[1] = x0
    x_traj[end] = xf
    x_traj
end

# problem
cost_fun = LQRCost(Q,R,Qf,xf)
obj = Objective(cost_fun,N)

goal_con = goal_constraint(xf)
bnd_con = BoundConstraint(n,m,u_min=u_min,u_max=u_max,trim=true)
con = ProblemConstraints([bnd_con],N)

prob = TrajectoryOptimization.Problem(model, obj,constraints=con, N=N, tf=tf0, x0=x0, dt=dt)
prob.constraints[N] += goal_con
copyto!(prob.X,line_trajectory(x0,xf,N))


X = line_trajectory(x0,xf,N)
U = [0.01*rand(m) for k = 1:N-1]
H = [dt*ones(1) for k = 1:N-1]

# dynamics constraints
# fc(z) = pendulum_dynamics_uncertain(z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
# fc(x,u,w) = pendulum_dynamics_uncertain(x,u,w)
fc(z) = let
    ż = zeros(eltype(z),n)
    prob.model.f(ż,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
    return ż
end

fc(x,u,w) = let
    ẋ = zero(x)
    prob.model.f(ẋ,x,u,w)
    return ẋ
end
∇fc(z) = ForwardDiff.jacobian(fc,z)
∇fc(x,u,w) = ∇fc([x;u;w])
dfcdx(x,u,w) = ∇fc(x,u,w)[:,1:n]
dfcdu(x,u,w) = ∇fc(x,u,w)[:,n .+ (1:m)]
dfcdw(x,u,w) = ∇fc(x,u,w)[:,(n+m) .+ (1:r)]

# pendulum_dynamics_uncertain(rand(n),rand(m),rand(r))
# fc([rand(n);rand(m);rand(r)])
# ∇fc([rand(n);rand(m);rand(r)])
# ∇fc(rand(n),rand(m),rand(r))
# dfcdx(rand(n),rand(m),rand(r))
# dfcdu(rand(n),rand(m),rand(r))
# dfcdw(rand(n),rand(m),rand(r))

xm(y,x,u,h) = 0.5*(y + x) + h/8*(fc(x,u,zeros(r)) - fc(y,u,zeros(r)))
dxmdy(y,x,u,h) = 0.5*I - h/8*dfcdx(y,u,zeros(r))
dxmdx(y,x,u,h) = 0.5*I + h/8*dfcdx(x,u,zeros(r))
dxmdu(y,x,u,h) = h/8*(dfcdu(x,u,zeros(r)) - dfcdu(y,u,zeros(r)))
dxmdh(y,x,u,h) = 1/8*(fc(x,u,zeros(r)) - fc(y,u,zeros(r)))

F(y,x,u,h) = y - x - h/6*fc(x,u,zeros(r)) - 4*h/6*fc(xm(y,x,u,h),u,zeros(r)) - h/6*fc(y,u,zeros(r))
F(z) = F(z[1:n],z[n .+ (1:n)],z[(n+n) .+ (1:m)],z[n+n+m+1])
∇F(Z) = ForwardDiff.jacobian(F,Z)
dFdy(y,x,u,h) = I - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdy(y,x,u,h) - h/6*dfcdx(y,u,zeros(r))
dFdx(y,x,u,h) = -I - h/6*dfcdx(x,u,zeros(r)) - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdx(y,x,u,h)
dFdu(y,x,u,h) = -h/6*dfcdu(x,u,zeros(r)) - 4*h/6*(dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdu(y,x,u,h) + dfcdu(xm(y,x,u,h),u,zeros(r))) - h/6*dfcdu(y,u,zeros(r))
dFdh(y,x,u,h) = -1/6*fc(x,u,zeros(r)) - 4/6*fc(xm(y,x,u,h),u,zeros(r)) - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdh(y,x,u,h) - 1/6*fc(y,u,zeros(r))
dFdw(y,x,u,h) = -h/6*dfcdw(x,u,zeros(r)) - 4*h/6*dfcdw(xm(y,x,u,h),u,zeros(r)) - h/6*dfcdw(y,u,zeros(r))

# y = rand(n); x = rand(n); u = rand(m); h = 0.1
#
# F(y,x,u,h)
# FF = ∇F([y;x;u;h])
# dFdy(y,x,u,h) == FF[:,1:n]
# isapprox(dFdx(y,x,u,h),FF[:,n .+ (1:n)])
# dFdu(y,x,u,h) == FF[:,(n+n) .+ (1:m)]
# isapprox(dFdh(y,x,u,h),FF[:,n+n+m+1])

function packZ(X,U,H)
    Z = [X[1];U[1];H[1]]

    for k = 2:N-1
        append!(Z,[X[k];U[k];H[k]])
    end
    append!(Z,X[N])

    return Z
end

ZZ = packZ(X,U,H)


unpackZ(Z) = let n=n, m=m, N=N, NN=NN
    X = [k != N ? Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][1:n] : Z[((k-1)*(n+m+1) + 1):NN] for k = 1:N]
    U = [Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][n .+ (1:m)] for k = 1:N-1]
    H = [Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][(n+m+1)] for k = 1:N-1]
    return X,U,H
end

unpackZ_timestep(Z) = let n=n, m=m, N=N, NN=NN
    Z_traj = [k != N ? Z[((k-1)*(n+m+1) + 1):k*(n+m+1)] : Z[((k-1)*(n+m+1) + 1):NN] for k = 1:N]
end

ℓ(x,u) = let Q=Q, R=R
    (x'*Q*x + u'*R*u)
end

dℓdx(x,u) = let Q=Q
    2*Q*x
end

dℓdu(x,u) = let R=R
    2*R*u
end

gf(x) = let Qf=Qf
    x'*Qf*x
end

dgfdx(x) = let Qf=Qf
    2*Qf*x
end

g(y,x,u,h) = let
    h/6*ℓ(x,u) + 4*h/6*ℓ(xm(y,x,u,h),u) + h/6*ℓ(y,u) + h
end



g(z) = g(z[1:n],z[n .+ (1:n)],z[(n+n) .+ (1:m)],z[n+n+m+1])
∇g(z) = ForwardDiff.gradient(g,z)
# g(y,x,u,h)
# gg = ∇g([y;x;u;h])

dgdx(y,x,u,h) = vec(h/6*dℓdx(x,u)' + 4*h/6*dℓdx(xm(y,x,u,h),u)'*dxmdx(y,x,u,h))
dgdy(y,x,u,h) = vec(4*h/6*dℓdx(xm(y,x,u,h),u)'*dxmdy(y,x,u,h) + h/6*dℓdx(y,u)')
dgdu(y,x,u,h) = vec(h/6*dℓdu(x,u)' + 4*h/6*(dℓdx(xm(y,x,u,h),u)'*dxmdu(y,x,u,h) + dℓdu(xm(y,x,u,h),u)') + h/6*dℓdu(y,u)')
dgdh(y,x,u,h) = 1/6*ℓ(x,u) + 4/6*ℓ(xm(y,x,u,h),u) + 4*h/6*dℓdx(xm(y,x,u,h),u)'*dxmdh(y,x,u,h) + 1/6*ℓ(y,u) + 1.0

# isapprox(dgdy(y,x,u,h),gg[1:n])
# isapprox(dgdx(y,x,u,h),gg[n .+ (1:n)])
# isapprox(dgdu(y,x,u,h),gg[(n+n) .+ (1:m)])
# isapprox(dgdh(y,x,u,h),gg[(n+n+m+1)])

robust_cost(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D
    X,U,H = unpackZ(Z)
    Ad = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdx(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
    Bd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdu(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
    Gd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdw(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]

    P = [zeros(eltype(Z),n,n) for k = 1:N]
    K = [zeros(eltype(Z),m,n) for k = 1:N-1]

    E = [zeros(eltype(Z),n,n) for k = 1:N]
    HH = [zeros(eltype(Z),n,r) for k = 1:N]

    P[N] = Qf_lqr
    E[1] = E0
    HH[1] = H0

    for k = N-1:-1:1
        K[k] = (R_lqr + Bd[k]'*P[k+1]*Bd[k])\(Bd[k]'*P[k+1]*Ad[k])
        P[k] = Q_lqr + K[k]'*R_lqr*K[k] + (Ad[k] - Bd[k]*K[k])'*P[k+1]*(Ad[k] - Bd[k]*K[k])
    end

    ℓ = 0.
    for k = 1:N-1
        Acl = Ad[k] - Bd[k]*K[k]
        E[k+1] = Acl*E[k]*Acl' + Gd[k]*HH[k]'*Acl' + Acl*HH[k]*Gd[k]' + Gd[k]*D*Gd[k]'
        HH[k+1] = Acl*HH[k] + Gd[k]*D
        ℓ += tr((Q_r + K[k]'*R_r*K[k])*E[k])
    end
    ℓ += tr(Qf_r*E[N])

    return ℓ
end

# robust_cost(ZZ)
∇robust_cost(Z) = ForwardDiff.gradient(robust_cost,Z)
# ∇robust_cost(ZZ)

cost(Z) = let n=n, m=m, N=N, NN=NN, Q=Q, R=R, Qf=Qf
    J = 0.
    Z_traj = unpackZ_timestep(Z)

    for k = 1:N-1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]
        J += g(y,x,u,h)
    end
    xN = Z_traj[N][1:n]
    J += gf(xN)

    return J
end

# cost(ZZ)

∇cost!(∇J,Z) = let n=n, m=m, N=N, NN=NN, Q=Q, R=R, Qf=Qf
    Z_traj = unpackZ_timestep(Z)
    k = 1
    y = Z_traj[k+1][1:n]
    x = Z_traj[k][1:n]
    u = Z_traj[k][n .+ (1:m)]
    h = Z_traj[k][n+m+1]
    xm = 0.5*(y+x)
    ∇J[1:(n+m+1+n)] = [dgdx(y,x,u,h);dgdu(y,x,u,h);dgdh(y,x,u,h);dgdx(y,x,u,h)]

    for k = 2:N-1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]
        xm = 0.5*(y+x)
        ∇J[((k-1)*(n+m+1) + 1):(k*(n+m+1)+n)] = [dgdx(y,x,u,h);dgdu(y,x,u,h);dgdh(y,x,u,h);dgdx(y,x,u,h)]
    end
    xN = Z_traj[N][1:n]
    ∇J[(NN - (n-1)):NN] = dgfdx(xN)

    return nothing
end

gen_E(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D
    X,U,H = unpackZ(Z)
    Ad = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdx(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
    Bd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdu(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
    Gd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdw(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]

    P = [zeros(eltype(Z),n,n) for k = 1:N]
    K = [zeros(eltype(Z),m,n) for k = 1:N-1]

    E = [zeros(eltype(Z),n,n) for k = 1:N]
    HH = [zeros(eltype(Z),n,r) for k = 1:N]
    E_vec = zeros(eltype(Z),N*n^2)

    P[N] = Qf_lqr
    E[1] = E0
    HH[1] = H0
    E_vec[1:n^2] = E0

    for k = N-1:-1:1
        K[k] = (R_lqr + Bd[k]'*P[k+1]*Bd[k])\(Bd[k]'*P[k+1]*Ad[k])
        P[k] = Q_lqr + K[k]'*R_lqr*K[k] + (Ad[k] - Bd[k]*K[k])'*P[k+1]*(Ad[k] - Bd[k]*K[k])
    end

    for k = 1:N-1
        Acl = Ad[k] - Bd[k]*K[k]
        E[k+1] = Acl*E[k]*Acl' + Gd[k]*HH[k]'*Acl' + Acl*HH[k]*Gd[k]' + Gd[k]*D*Gd[k]'
        HH[k+1] = Acl*HH[k] + Gd[k]*D
        E_vec[k*(n^2) .+ (1:n^2)] = vec(E[k+1])
    end

    return E_vec
end

gen_δx(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D, eig_thr=eig_thr
    E = gen_E(Z)
    Esqrt = zero(E)

    for k = 1:N
        ee = reshape(E[(k-1)*n^2 .+ (1:n^2)],n,n)

        # make E >= 0
        if !isposdef(ee)
            _eig = eigen(ee)
            _eig.values .= real.(_eig.values)
            for i = 1:n
                if _eig.values[i] < 0.
                    _eig.values[i] = eig_thr
                end
            end
            Esqrt[(k-1)*n^2 .+ (1:n^2)] = Diagonal(sqrt.(_eig.values))*_eig.vectors'
        else
            Esqrt[(k-1)*n^2 .+ (1:n^2)] = real.(vec(sqrt(ee)))
        end
    end

    return Esqrt
end

# gen_δx(ZZ)

dEdZ(Z) = ForwardDiff.jacobian(gen_E,Z)

dδxdZ(Z) = let n=n,m=m,N=N,NN=NN
    res = zeros(eltype(Z),N*n^2,NN)
    Esqrt = gen_δx(Z)
    ∇E = dEdZ(Z)
    In = Diagonal(ones(n))

    for k = 1:N
        idx = (k-1)*n^2 .+ (1:n^2)
        _Esqrt = reshape(Esqrt[idx],n,n)
        res[idx,1:NN] = pinv(kron(In,_Esqrt) + kron(_Esqrt',In))*∇E[idx,1:NN]
        # res[idx,1:NN] = (kron(In,_Esqrt) + kron(_Esqrt',In))\∇E[idx,1:NN]
    end

    res
end

gen_KEK(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D
    X,U,H = unpackZ(Z)
    Ad = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdx(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
    Bd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdu(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]
    Gd = [-dFdy(X[k+1],X[k],U[k],H[k])\dFdw(X[k+1],X[k],U[k],H[k]) for k = 1:N-1]

    P = [zeros(eltype(Z),n,n) for k = 1:N]
    K = [zeros(eltype(Z),m,n) for k = 1:N-1]

    E = [zeros(eltype(Z),n,n) for k = 1:N]
    HH = [zeros(eltype(Z),n,r) for k = 1:N]
    KEK_vec = zeros(eltype(Z),(N-1)*m^2)

    P[N] = Qf_lqr
    E[1] = E0
    HH[1] = H0

    for k = N-1:-1:1
        K[k] = (R_lqr + Bd[k]'*P[k+1]*Bd[k])\(Bd[k]'*P[k+1]*Ad[k])
        P[k] = Q_lqr + K[k]'*R_lqr*K[k] + (Ad[k] - Bd[k]*K[k])'*P[k+1]*(Ad[k] - Bd[k]*K[k])
    end

    for k = 1:N-1
        KEK_vec[(k-1)*(m^2) .+ (1:m^2)] = vec(K[k]*E[k]*K[k]')

        Acl = Ad[k] - Bd[k]*K[k]
        E[k+1] = Acl*E[k]*Acl' + Gd[k]*HH[k]'*Acl' + Acl*HH[k]*Gd[k]' + Gd[k]*D*Gd[k]'
        HH[k+1] = Acl*HH[k] + Gd[k]*D
    end

    return KEK_vec
end

gen_δu(Z) = let n=n, m=m,r=r, N=N, NN=NN, Q_lqr=Q_lqr, R_lqr=R_lqr, Qf_lqr=Qf_lqr, E0=E0, H0=H0, D=D, eig_thr=eig_thr
    KEK = gen_KEK(Z)
    KEKsqrt = zero(KEK)

    for k = 1:N-1
        kek = reshape(KEK[(k-1)*m^2 .+ (1:m^2)],m,m)

        if !isposdef(kek)
            _eig = eigen(kek)
            _eig.values .= real.(_eig.values)
            for i = 1:m
                if _eig.values[i] < 0.
                    _eig.values[i] = eig_thr
                end
            end
            KEKsqrt[(k-1)*m^2 .+ (1:m^2)] = Diagonal(sqrt.(_eig.values))*_eig.vectors'
        else
            KEKsqrt[(k-1)*m^2 .+ (1:m^2)] = real.(vec(sqrt(kek)))
        end

    end

    return KEKsqrt
end

# gen_δu(ZZ)

# gen_KEK(ZZ)

dKEKdZ(Z) = ForwardDiff.jacobian(gen_KEK,Z)

dδudZ(Z) = let n=n,m=m,N=N,NN=NN
    res = zeros(eltype(Z),(N-1)*m^2,NN)
    ∇KEK = dKEKdZ(Z)
    KEKsqrt = gen_δu(Z)
    Im = Diagonal(ones(m))

    for k = 1:N-1
        idx = (k-1)*m^2 .+ (1:m^2)
        _KEKsqrt = reshape(KEKsqrt[idx],m,m)

        res[idx,1:NN] = (kron(Im,_KEKsqrt) + kron(_KEKsqrt',Im))\∇KEK[idx,1:NN]
    end

    res
end

# dδudZ(ZZ)

dynamics_constraints!(g,Z) = let n=n, m=m, N=N, NN=NN, x0=x0, xf=xf
    Z_traj = unpackZ_timestep(Z)
    k = 1
    y = Z_traj[k+1][1:n]
    x = Z_traj[k][1:n]
    u = Z_traj[k][n .+ (1:m)]
    h = Z_traj[k][n+m+1]
    con = x-x0

    for k = 1:N-1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]
        append!(con,F(y,x,u,h))

        if k != N-1
            h₊ = Z_traj[k+1][n+m+1]
            append!(con,h - h₊)
        end
    end
    xN = Z_traj[N][1:n]
    append!(con,xN-xf)

    copyto!(g,con)
    return nothing
end

∇dynamics_constraints!(∇con,Z) = let n=n, m=m, N=N, NN=NN, x0=x0, xf=xf
    # ∇con = zeros(p,NN)
    Z_traj = unpackZ_timestep(Z)

    shift = 0
    # con = x-x0
    copyto!(view(∇con,1:n^2),vec(Diagonal(ones(n))))
    shift += n^2

    for k = 1:N-1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]

        # dynamics constraint jacobians
        copyto!(view(∇con,shift .+ (1:(n+m+1+n)*(n))),vec([dFdx(y,x,u,h) dFdu(y,x,u,h) dFdh(y,x,u,h) dFdy(y,x,u,h)]))
        shift += (n+m+1+n)*n

        if k != N-1
            copyto!(view(∇con,shift .+ (1:2)),[1.;-1.])
            shift += 2
        end
    end
    k = N
    xN = Z_traj[k][1:n]
    copyto!(view(∇con,shift .+ (1:n^2)),vec(Diagonal(ones(n))))
    shift += n^2

    return nothing
end

function add_rows_cols!(row,col,_r,_c)
    for cc in _c
        for rr in _r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
end

dynamics_constraints_sparsity() = let n=n, m=m, N=N, NN=NN, x0=x0, xf=xf

    row = []
    col = []

    r = 1:n
    c = 1:n

    add_rows_cols!(row,col,r,c)

    for k = 1:N-1
        # dynamics constraint jacobians
        c_idx = ((k-1)*(n+m+1)) .+ (1:(n+m+1+n))
        r_idx = (n + (k-1)*(n + 1)) .+ (1:n)

        add_rows_cols!(row,col,r_idx,c_idx)

        if k != N-1
            c_idx = ((k-1)*(n+m+1) + n + m) .+ 1
            c2_idx = ((k)*(n+m+1) + n + m) .+ 1
            r_idx = (n + n + (k-1)*(n + 1)) .+ 1

            add_rows_cols!(row,col,r_idx,c_idx)

            add_rows_cols!(row,col,r_idx,c2_idx)
        end
    end

    k = N
    r = ((n + (k-1)*(n+1) - 1) .+ (1:n))
    c = (NN - (n-1)):NN

    add_rows_cols!(row,col,r,c)

    return collect(zip(row,col))
end

function num_robust_constraints(prob::TrajectoryOptimization.Problem)
    n = prob.model.n; m = prob.model.m; N = prob.N

    p = 0
    p_jac = 0
    for k = 1:N-1
        con_ineq, con_eq = split(prob.constraints[k])
        for con in con_ineq
            p_con = length(con)
            # p += p_con*((2*n+1)*(2*m+1) - 1)
            # p_jac += p_con*((2*n+1)*(2*m+1) - 1)*NN
            p += p_con*(2*m)
            p_jac += p_con*(2*m)*NN
        end
    end

    # con_ineq, con_eq = split(prob.constraints[N])
    # for con in con_ineq
    #     if con isa BoundConstraint
    #         p_con = sum(con.active.x_max + con.active.x_min)
    #     else
    #         p_con = length(con)
    #     end
    #     p += p_con*(2*n)
    #     p_jac += p_con*(2*n)*NN
    # end

    return p,p_jac
end

robust_constraints!(g,Z) = let n=n, m=m, N=N, NN=NN
    pp = 0
    shift = 0
    δx = gen_δx(Z)
    δu = gen_δu(Z)

    X,U,H = unpackZ(Z)

    for k = 1:N-1
        δx_k = δx[(k-1)*n^2 .+ (1:n^2)]
        Xr = [X[k]]
        δx_sign = [0]
        # for i = 1:n
        #     δx_k_i = δx_k[(i-1)*n .+ (1:n)]
        #     push!(Xr,X[k]+δx_k_i)
        #     push!(Xr,X[k]-δx_k_i)
        #
        #     push!(δx_sign,1)
        #     push!(δx_sign,-1)
        # end

        δu_k = δu[(k-1)*m^2 .+ (1:m^2)]
        Ur = [U[k]]
        δu_sign = [0]
        for j = 1:m
            δu_k_j = δu_k[(j-1)*m .+ (1:m)]
            push!(Ur,U[k]+δu_k_j)
            push!(Ur,U[k]-δu_k_j)

            push!(δu_sign,1)
            push!(δu_sign,-1)
        end

        con_ineq, con_eq = split(prob.constraints[k])
        for con in con_ineq
            p_con = length(con)

            for (i,xr) in enumerate(Xr)
                for (j,ur) in enumerate(Ur)
                    if !(j == i && (i == 1 || j == 1))
                        # println("$i,$j")
                        if p_con != 0
                            evaluate!(view(g,shift .+ (1:p_con)),con,xr,ur)
                            shift += p_con
                        end
                    end
                end
            end
        end
    end

    # k = N
    # con_ineq, con_eq = split(prob.constraints[N])
    # for con in con_ineq
    #     if con isa BoundConstraint
    #         p_con = sum(con.active.x_max + con.active.x_min)
    #     else
    #         p_con = length(con)
    #     end
    #     pp += p_con*(2*n)
    #     δx_N = δx[(k-1)*n^2 .+ (1:n^2)]
    #     Xr = []
    #     δxN_sign = []
    #     for i = 1:n
    #         δx_N_i = δx_N[(i-1)*n .+ (1:n)]
    #         push!(Xr,X[k]+δx_N_i)
    #         push!(Xr,X[k]-δx_N_i)
    #
    #         push!(δxN_sign,1)
    #         push!(δxN_sign,-1)
    #     end
    #
    #     for (i,xr) in enumerate(Xr)
    #         if p_con != 0
    #             evaluate!(view(g,shift .+ (1:p_con)),con,xr)
    #             shift += p_con
    #         end
    #     end
    # end

    return nothing
end

p_robust, p_robust_jac = num_robust_constraints(prob)
gg_robust = zeros(p_robust)
robust_constraints!(gg_robust,ZZ)
gg_robust


∇robust_constraints!(G,Z) = let n=n, m=m, N=N, NN=NN
    pp = 0
    shift = 0
    δx = gen_δx(Z)
    δu = gen_δu(Z)

    ∇δx = dδxdZ(Z)
    ∇δu = dδudZ(Z)

    X,U,H = unpackZ(Z)

    In = Diagonal(ones(n))
    Im = Diagonal(ones(m))

    for k = 1:N-1
        ∇δx_k = ∇δx[(k-1)*n^2 .+ (1:n^2),1:NN]
        ∇δu_k = ∇δu[(k-1)*m^2 .+ (1:m^2),1:NN]

        x_idx = (k-1)*(n+m+1) .+ (1:n)
        u_idx = ((k-1)*(n+m+1) + n) .+ (1:m)

        δx_k = δx[(k-1)*n^2 .+ (1:n^2)]
        Xr = [X[k]]
        δx_sign = [(0,0)] # (sign,column index)
        # for i = 1:n
        #     δx_k_i = δx_k[(i-1)*n .+ (1:n)]
        #     push!(Xr,X[k]+δx_k_i)
        #     push!(Xr,X[k]-δx_k_i)
        #
        #     push!(δx_sign,(1,i))
        #     push!(δx_sign,(-1,i))
        # end

        δu_k = δu[(k-1)*m^2 .+ (1:m^2)]
        Ur = [U[k]]
        δu_sign = [(0,0)]
        for j = 1:m
            δu_k_j = δu_k[(j-1)*m .+ (1:m)]
            push!(Ur,U[k]+δu_k_j)
            push!(Ur,U[k]-δu_k_j)

            push!(δu_sign,(1,j))
            push!(δu_sign,(-1,j))
        end

        con_ineq, con_eq = split(prob.constraints[k])
        for con in con_ineq
            p_con = length(con)

            pp += p_con*((2*n+1)*(2*m+1) - 1)

            C = zeros(p_con,n+m)
            jacobian!(C,con,X[k],U[k])
            Cx = C[:,1:n]

            # @assert Cx == [0. 0.; 0. 0.]

            Cu = C[:,n .+ (1:m)]

            # @assert Cu == [1.0;-1.0]

            for (i,xr) in enumerate(Xr)
                for (j,ur) in enumerate(Ur)
                    if !(j == i && (i == 1 || j == 1))
                        # println("$i,$j")
                        if p_con != 0
                            if i != 1
                                ∇δx_k_i = δx_sign[i][1]*copy(∇δx_k[(δx_sign[i][2]-1)*n .+ (1:n),1:NN])
                            else # no δx
                                ∇δx_k_i = zeros(n,NN)
                            end

                            if j != 1
                                ∇δu_k_j = δu_sign[j][1]*copy(∇δu_k[(δu_sign[j][2]-1)*m .+ (1:m),1:NN])
                            else # no δu
                                ∇δu_k_j = zeros(m,NN)
                            end

                            ∇δx_k_i[1:n,x_idx] += In
                            ∇δu_k_j[1:m,u_idx] += Im

                            tmp = Cu*∇δu_k_j + Cx*∇δx_k_i

                            G[shift .+ (1:p_con*NN)] = vec(tmp)
                            shift += p_con*NN
                        end
                    end
                end
            end
        end
    end

    # k = N
    # con_ineq, con_eq = split(prob.constraints[N])
    # for con in con_ineq
    #     if con isa BoundConstraint
    #         p_con = sum(con.active.x_max + con.active.x_min)
    #     else
    #         p_con = length(con)
    #     end
    #     pp += p_con*(2*n)
    #     δx_N = δx[(k-1)*n^2 .+ (1:n^2)]
    #     Xr = []
    #     δxN_sign = []
    #     for i = 1:n
    #         δx_N_i = δx_N[(i-1)*n .+ (1:n)]
    #         push!(Xr,X[k]+δx_N_i)
    #         push!(Xr,X[k]-δx_N_i)
    #
    #         push!(δxN_sign,(1,i))
    #         push!(δxN_sign,(-1,i))
    #     end
    #
    #     CxN = zeros(p_con,n)
    #     jacobian!(CxN,con,X[N])
    #
    #     for (i,xr) in enumerate(Xr)
    #         if p_con != 0
    #             ∇δx_N_i = copy(∇δx_N[(δxN_sign[i][2]-1)*n .+ (1:n),1:NN])
    #             ∇δx_N_i[1:n,x_idx] += δxN_sign[i][1]*In
    #
    #             G[shift .+ (1:p_con*NN)] = vec(Cx*∇δx_N_i)
    #             shift += p_con*NN
    #         end
    #     end
    # end


    return nothing
end

robust_constraint_sparsity(shift=0) = let n=n, m=m, N=N, NN=NN
    idx_col = 1:NN

    row = []
    col = []

    for k = 1:N-1


        con_ineq, con_eq = split(prob.constraints[k])
        for con in con_ineq
            p_con = length(con)

            for i = 1:1#(2*n + 1)
                for j = 1:(2*m + 1)
                    if !(j == i && (i == 1 || j == 1))
                        # println("$i,$j")
                        if p_con != 0
                            idx_row = shift .+ (1:p_con)
                            add_rows_cols!(row,col,idx_row,idx_col)
                            shift += p_con
                        end
                    end
                end
            end
        end
    end

    # k = N
    # con_ineq, con_eq = split(prob.constraints[N])
    # for con in con_ineq
    #     if con isa BoundConstraint
    #         p_con = sum(con.active.x_max + con.active.x_min)
    #     else
    #         p_con = length(con)
    #     end
    #
    #     for i = 1:2*n
    #         if p_con != 0
    #
    #             idx_row = shift .+ (1:p_con)
    #             add_rows_cols!(row,col,idx_row,idx_col)
    #             shift += p_con
    #         end
    #     end
    # end

    return collect(zip(row,col))
end

p_robust,p_robust_jac = num_robust_constraints(prob)
∇robust_constraints!(zeros(p_robust_jac),ZZ)
robust_constraint_sparsity(p_dynamics)

robust_constraint_bounds(prob) = let n=n, m=m, N=N, NN=NN
    p, = num_robust_constraints(prob)
    bL = -Inf*ones(p)
    bU = zeros(p)

    return bL,bU
end

primal_bounds() = let n=n, m=m, N=N, NN=NN, u_max=u_max, u_min=u_min, h_max=h_max, h_min=h_min

    Z_low = [-Inf*ones(n);u_min*ones(m);h_min]
    Z_up = [Inf*ones(n);u_max*ones(m);h_max]

    for k = 2:N-1
        append!(Z_low,[-Inf*ones(n);u_min*ones(m);h_min])
        append!(Z_up,[Inf*ones(n);u_max*ones(m);h_max])
    end
    append!(Z_low,-Inf*ones(n))
    append!(Z_up,Inf*ones(n))

    return Z_low, Z_up
end

dynamics_constraint_bounds() = let p=p_dynamics
    ceq_low = zeros(p); ceq_up = zeros(p)

    return ceq_low,ceq_up
end

# constraint_bounds()

## set up optimization (MathOptInterface)
ZZ = packZ(X,U,H)
ZZ_traj = unpackZ_timestep(ZZ)
XX,UU,HH = unpackZ(ZZ)

p_dynamics_jac = n*(N-1)*(n+m+1+n) + 2*n^2 + (N-2)*2

idx = dynamics_constraints_sparsity()
idx_robust = robust_constraint_sparsity(p_dynamics)

append!(idx,idx_robust)

struct DIRTRELProblem <: MOI.AbstractNLPEvaluator
    enable_hessian::Bool
end

MOI.features_available(pr::DIRTRELProblem) = [:Grad, :Jac]
MOI.initialize(pr::DIRTRELProblem, features) = nothing
MOI.jacobian_structure(pr::DIRTRELProblem) = idx
MOI.hessian_lagrangian_structure(pr::DIRTRELProblem) = []

function MOI.eval_objective(pr::DIRTRELProblem, Z)
    return cost(Z) + robust_cost(Z)
end

function MOI.eval_objective_gradient(pr::DIRTRELProblem, grad_f, Z)
    ∇cost!(grad_f, Z)
    grad_f + ∇robust_cost(Z)
end

function MOI.eval_constraint(pr::DIRTRELProblem, g, Z)
    g_dynamics = view(g,1:p_dynamics)
    g_robust = view(g,p_dynamics .+ (1:p_robust))
    dynamics_constraints!(g_dynamics,Z)
    robust_constraints!(g_robust,Z)
end

function MOI.eval_constraint_jacobian(pr::DIRTRELProblem, jac, Z)
    jac_dynamics = view(jac,1:p_dynamics_jac)
    jac_robust = view(jac,p_dynamics_jac .+ (1:p_robust_jac))
    ∇dynamics_constraints!(jac_dynamics,Z)
    ∇robust_constraints!(jac_robust,Z)
end

MOI.eval_hessian_lagrangian(pr::DIRTRELProblem, H, x, σ, μ) = nothing

pr = DIRTRELProblem(false)

Z_low, Z_up = primal_bounds()
c_low, c_up = dynamics_constraint_bounds()
r_low, r_up = robust_constraint_bounds(prob)

g_low = [c_low;r_low]; g_up = [c_up;r_up]

nlp_bounds = MOI.NLPBoundsPair.(g_low,g_up)
block_data = MOI.NLPBlockData(nlp_bounds,pr,true)

solver = Ipopt.Optimizer(tol=1.0e-3,constr_viol_tol=1.0e-2)
solver
Z = MOI.add_variables(solver,NN)

for i = 1:NN
    zi = MOI.SingleVariable(Z[i])
    MOI.add_constraint(solver, zi, MOI.LessThan(Z_up[i]))
    MOI.add_constraint(solver, zi, MOI.GreaterThan(Z_low[i]))
    MOI.set(solver, MOI.VariablePrimalStart(), Z[i], ZZ[i])
end

# Solve the problem
MOI.set(solver, MOI.NLPBlock(), block_data)
MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
MOI.optimize!(solver)

# Get the solution
res = MOI.get(solver, MOI.VariablePrimal(), Z)

X,U,H = unpackZ(res)
Xblk = zeros(n,N)
for k = 1:N
    Xblk[:,k] = X[k]
end
cost(ZZ)
cost(res)

Ublk = zeros(m,N-1)
for k = 1:N-1
    Ublk[:,k] = U[k]
end

# t_run = zeros(N)
# t_run[1] = 0
# for k = 1:N-1
#     t_run[k+1] = t_run[k] + H[k][1]
# end
#
# u_run = zeros(N)
# for k = 1:N-1
#     u_run[k] = U[k][1]
# end
# u_run[N] = U[end][1]
#
# plot(t_run,u_run)


plot(Xblk')
plot(Ublk')

idx

struct DIRTRELProblem <: MOI.AbstractNLPEvaluator
    prob::Problem
    p_dynamics::Int
    p_dynamics_jac::Int
    p_custom::Int
    p_custom_jac::Int
    p_robust::Int
    p_robust_jac::Int
    jac_struct::Tuple

    enable_hessian::Bool
end

@with_kw mutable struct DIRTRELSolverOptions{T} <: DirectSolverOptions{T}
    E0::AbstractMatrix
    H0::AbstractMatrix
    D::AbstractMatrix

    Q_lqr::AbstractMatrix
    R_lqr::AbstractMatrix
    Qf_lqr::AbstractMatrix

    Q_r::AbstractMatrix
    R_r::AbstractMatrix
    Qf_r::AbstractMatrix

    integration::Symbol = :HS # Hermite-Simpson (zoh on control)

    "NLP Solver to use. Options are (:Ipopt) (more to be added in the future)"
    nlp::Symbol = :Ipopt

    "Options dictionary for the nlp solver"
    opts::Dict{String,Any} = Dict{String,Any}()

    "Print output to console"
    verbose::Bool = true
end
