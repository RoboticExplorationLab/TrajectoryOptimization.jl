using Plots, ForwardDiff

model = Dynamics.pendulum_model
# model = Dynamics.doubleintegrator_model
# model = Dynamics.car_model
fc = model.f
n = model.n; m = model.m

x0 = [0.; 0.]
xf = [pi; 0.]

# x0 = [0.; 0.]
# xf = [1.0; 0.]

# x0 = [0.;0.;0.]
# xf = [0.;1.;0.]
#
Q = 1.0*Diagonal(ones(n))
R = 1.0*Diagonal(ones(m))
Qf = 1000.0*Diagonal(ones(n))

N = 201
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
    fc1 = fc2 = fc3 = zero(X[k])

    copyto!(X[k+1], X[k])
    g = Inf

    cnt = 0
    while norm(g) > 1.0e-12
        cnt += 1
        if cnt > 10
            error("Integration convergence fail")
        end

        fc(fc1,X[k],U[k])
        fc(fc3,X[k+1],U[k])
        Xm = 0.5*(X[k] + X[k+1]) + dt/8*(fc1 - fc3)
        fc(fc2,Xm,U[k])


        g = X[k+1] - X[k] - dt/6*fc1 - 4/6*dt*fc2 - dt/6*fc3

        A1 = ∇f(Xm,U[k])[:,1:n]
        A2 = ∇f(X[k+1],U[k])[:,1:n]
        ∇g = Diagonal(I,n) - 4/6*dt*A1*(0.5*Diagonal(I,n) - dt/8*A2) - dt/6*A2
        δx = -∇g\g

        X[k+1] += δx
    end
end
plot(X)

X_for = deepcopy(X)

X = [zeros(n) for k = 1:N]
X[N] = copy(X_for[end])

# Implicit backward dynamics
for k = N-1:-1:1
    # println(k)
    fc1 = fc2 = fc3 = zero(X[k])

    copyto!(X[k], X[k+1])
    g = Inf

    cnt = 0
    while norm(g) > 1.0e-12
        cnt += 1

        if cnt > 10
            error("Integration convergence fail")
        end

        fc(fc1,X[k+1],U[k])
        fc(fc3,X[k],U[k])

        Xm = 0.5*(X[k+1] + X[k]) - dt/8*(fc1 - fc3)
        fc(fc2,Xm,U[k])

        g = X[k] - X[k+1] + dt/6*fc1 + 4/6*dt*fc2 + dt/6*fc3

        A1 = ∇f(Xm,U[k])[:,1:n]
        A2 = ∇f(X[k],U[k])[:,1:n]

        ∇g = Diagonal(I,n) + 4/6*dt*A1*(0.5*Diagonal(I,n) + dt/8*A2) + dt/6*A2
        δx = -∇g\g

        X[k] += δx
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

for k = N-1:-1:1
    println(k)
    s1 = s2 = s3 = zero(S[k])
    fc1 = fc2 = fc3 = zero(X[k])

    copyto!(S[k], S[k+1])
    g = Inf

    cnt = 0
    while norm(g) > 1.0e-12
        cnt += 1
        println(norm(g))

        gp = copy(g)
        if cnt > 1000
            error("Integration convergence fail")
        end
        riccati(s1,S[k+1],X[k+1],U[k])
        riccati(s3,S[k],X[k],U[k])
        fc(fc1,X[k+1],U[k])
        fc(fc3,X[k],U[k])

        Sm = 0.5*(S[k+1] + S[k]) - dt/8*(s1 - s3)
        Xm = 0.5*(X[k+1] + X[k]) - dt/8*(fc1 - fc3)
        riccati(s2,Sm,Xm,U[k])

        g = S[k] - S[k+1] + dt/6*s1 + 4/6*dt*s2 + dt/6*s3

        A1 = ∇riccati(Sm,Xm,U[k])[:,1:n^2]
        A2 = ∇riccati(S[k],X[k],U[k])[:,1:n^2]


        ∇g = Diagonal(I,n^2) + 4/6*dt*A1*(0.5*Diagonal(I,n^2) + dt/8*A2) + dt/6*A2
        δs = -∇g\g

        S[k] += δs

    end
end

# plot(S)
Ps = [vec(reshape(S[k],n,n)*reshape(S[k],n,n)') for k = 1:N]
Sb = copy(S)
S1 = cholesky(reshape(Sb[1],n,n)*reshape(Sb[1],n,n)').U
plot(Ps,legend=:left)

# Implicit midpoint Riccati forward
S = [zeros(n^2) for k = 1:N]
# X = prob.X
# U = prob.U

S[1] = vec(Sb[1])

for k = 1:N-1
    println(k)
    ss = s1 = s2 = s3 = zero(S[k])
    fc1 = fc2 = fc3 = zero(X[k])

    copyto!(S[k+1], S[k])
    g = Inf
    α = 1.
    cnt = 0
    while norm(g) > 1.0e-12
        cnt += 1
        println(norm(g))
        if cnt > 1000
            error("Integration convergence fail")
        end

        riccati(s1,S[k],X[k],U[k])
        riccati(s3,S[k+1],X[k+1],U[k])
        fc(fc1,X[k],U[k])
        fc(fc3,X[k+1],U[k])

        Sm = 0.5*(S[k] + S[k+1]) + dt/8*(s1 - s3)
        Xm = 0.5*(X[k] + X[k+1]) + dt/8*(fc1 - fc3)
        riccati(s2,Sm,Xm,U[k])

        A1 = ∇riccati(Sm,Xm,U[k])[:,1:n^2]
        A2 = ∇riccati(S[k+1],X[k+1],U[k])[:,1:n^2]

        g = S[k+1] - S[k] - dt/6*s1 - 4/6*dt*s2 - dt/6*s3
        ∇g = Diagonal(I,n^2) - 4/6*dt*A1*(0.5*Diagonal(I,n^2) - dt/8*A2) - dt/6*A2
        δs = -∇g\g

        # g_new = copy(g)
        # α = 2.
        # cnt_ls = 0
        # while norm(g_new) >= norm(g)
        #     α /= 2.
        #     cnt_ls += 1
        #     tmp = S[k+1] + α*δs
        #     riccati(s3,tmp,X[k+1],U[k])
        #
        #     Sm = 0.5*(S[k] + tmp) + dt/8*(s1 - s3)
        #     riccati(s2,Sm,Xm,U[k])
        #     g_new = tmp - S[k] - dt/6*s1 - 4/6*dt*s2 - dt/6*s3
        #
        #     println(norm(g_new))
        #     if cnt_ls > 1000
        #         error("line search failed")
        #     end
        # end
        S[k+1] += α*δs
    end
end

Ps_for = [vec(reshape(S[k],n,n)*reshape(S[k],n,n)') for k = 1:N]

plot(Ps,legend=:left,color=:purple,label="backward")
plot!(Ps_for,legend=:left,color=:orange,label="forward",style=:dash)
