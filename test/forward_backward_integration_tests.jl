using Plots

model = Dynamics.doubleintegrator_model
fc = model.f
n = model.n; m = model.m

x0 = [0.; 0.]
xf = [1.0; 0.]

Q = 1.0*Diagonal(ones(n))
R = 1.0*Diagonal(ones(m))
Qf = 1000.0*Diagonal(ones(n))

N = 101
tf = 1.0
dt = tf/(N-1)

U = [ones(m) for k = 1:N-1]

# continous time riccati
function riccati(ṡ,s,t)
    S = reshape(s,n,n)
    ṡ .= vec(-.5*Q*inv(S') - _Ac(t)'*S + .5*(S*S'*_Bc(t))*(R\(_Bc(t)'*S)));
end

f(ẋ,z) = model.f(ẋ,z[1:n],z[n .+ (1:m)])
∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
∇f(x,u) = ∇f([x;u])

X = [zeros(n) for k = 1:N]
X[1] = copy(x0)

# Implicit forward dynamics
for k = 1:N-1
    # println(k)
    k1 = k2 = kg = zero(X[k])

    copyto!(X[k+1], X[k])# + k2)
    g = Inf
    gp = Inf
    α = 1.0
    cnt = 0
    while norm(g) > 1.0e-12
        cnt += 1
        if norm(g) > norm(gp)
            α *= 0.9
        else
            α = 1.0
        end
        gp = copy(g)
        Xm = 0.5*(X[k+1] + X[k])

        fc(kg,Xm,U[k])
        g = X[k+1] - X[k] - dt*kg

        A = ∇f(Xm,U[k])[:,1:n]

        ∇g = Diagonal(I,n) - 0.5*dt*A
        δs = -∇g\g

        X[k+1] += δs
    end
end
plot(X)

X_for = deepcopy(X)

X = [zeros(n) for k = 1:N]
X[N] = copy(X_for[end])

# Implicit backward dynamics
for k = N:-1:2
    # println(k)
    k1 = k2 = kg = zero(X[k])

    copyto!(X[k-1], X[k])# + k2)
    g = Inf
    gp = Inf
    α = 1.0
    cnt = 0
    while norm(g) > 1.0e-12
        cnt += 1
        if norm(g) > norm(gp)
            α *= 0.9
        else
            α = 1.0
        end
        gp = copy(g)
        Xm = 0.5*(X[k-1] + X[k])

        fc(kg,Xm,U[k-1])
        g = X[k-1] - X[k] + dt*kg

        A = ∇f(Xm,U[k-1])[:,1:n]

        ∇g = Diagonal(I,n) + 0.5*dt*A
        δs = -∇g\g

        X[k-1] += δs
    end
end

plot(X_for,color=:orange,label="forward")
plot!(X,color=:purple,style=:dash,label="backward")

function riccati(ṡ,s,x,u)
    S = reshape(s,n,n)
    F = ∇f(x,u)
    A = F[:,1:n]
    B = F[:,n .+ (1:m)]

    Si = inv(S')
    ṡ .= vec(-.5*Q*Si - A'*S + .5*(S*S'*B)*(R\(B'*S)))
end

riccati_wrap(ṡ,z) = riccati(ṡ,z[1:n^2],z[n^2 .+ (1:n)],z[(n^2 + n) .+ (1:m)])
riccati_wrap(rand(n^2),[rand(n^2);rand(n);rand(m)])

∇riccati(z) = ForwardDiff.jacobian(riccati_wrap,zeros(eltype(z),n^2),z)
∇riccati(s,x,u) = ∇riccati([s;x;u])

# implicit midpoint (reverse to solve Riccati backward)
S = [zeros(n^2) for k = 1:N]
# X = prob.X
# U = prob.U

S[N] = vec(cholesky(Qf).U)

for k = N:-1:2
    # println(k)
    k1 = k2 = kg = zero(S[k])
    # kk1 = zero(X[k])
    # riccati(k1, S[k], X[k], U[k-1])
    # fc(kk1,X[k],U[k-1])
    # k1 *= -dt
    # kk1 *= -dt
    #
    # riccati(k2, S[k] + k1/2, X[k] + kk1/2, U[k-1]);
    # k2 *= -dt

    copyto!(S[k-1], S[k])# + k2)
    g = Inf
    gp = Inf
    α = 1.0
    cnt = 0
    while norm(g) > 1.0e-12
        cnt += 1
        if norm(g) > norm(gp)
            α *= 0.9
        else
            α = 1.0
        end
        gp = copy(g)
        if cnt > 25
            error("Integration convergence fail")
        end
        Sm = 0.5*(S[k-1] + S[k])
        Xm = 0.5*(X[k-1] + X[k])

        riccati(kg,Sm,Xm,U[k-1])
        g = S[k-1] - S[k] + dt*kg

        A = ∇riccati(Sm,Xm,U[k-1])[:,1:n^2]

        ∇g = Diagonal(I,n^2) + 0.5*dt*A
        δs = -∇g\g

        S[k-1] += α*δs
    end
end

# plot(S)
Ps = [vec(reshape(S[k],n,n)*reshape(S[k],n,n)') for k = 1:N]
Sb = copy(S)

# Implicit midpoint Riccati forward
S = [zeros(n^2) for k = 1:N]
# X = prob.X
# U = prob.U

S[1] = vec(Sb[1])

for k = 1:N-1
    println(k)
    k1 = k2 = kg = zero(S[k])
    # kk1 = zero(X[k])
    # riccati(k1, S[k], X[k], U[k])
    # fc(kk1,X[k],U[k])
    # k1 *= dt
    # kk1 *= dt
    #
    # riccati(k2, S[k] + k1/2, X[k] + kk1/2, U[k]);
    # k2 *= dt

    copyto!(S[k+1], S[k])# + k2)
    g = Inf
    gp = Inf
    α = 1.0
    cnt = 0
    while norm(g) > 1.0e-12
        cnt += 1
        # if norm(g) > norm(gp)
        #     α *= 0.9
        # else
        #     α = 1.0#min(1.0,α*1.01)
        # end
        # gp = copy(g)
        println(norm(g))
        if cnt > 100
            error("Integration convergence fail")
        end

        Sm = 0.5*(S[k+1] + S[k])
        Xm = 0.5*(X[k+1] + X[k])

        riccati(kg,Sm,Xm,U[k])
        g = S[k+1] - S[k] - dt*kg

        A = ∇riccati(Sm,Xm,U[k])[:,1:n^2]

        ∇g = Diagonal(I,n^2) - 0.5*dt*A
        δs = -∇g\g

        S[k+1] += α*δs
    end
end

Ps_for = [vec(reshape(S[k],n,n)*reshape(S[k],n,n)') for k = 1:N]

plot(Ps,legend=:left,color=:purple,label="backward")
plot!(Ps_for,legend=:left,color=:orange,label="forward",style=:dash)
