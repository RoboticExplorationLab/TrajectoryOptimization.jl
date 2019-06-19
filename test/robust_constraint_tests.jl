using Test

model = Dynamics.car_model_uncertain
n = model.n; m = model.m; r = model.r

T = Float64

# costs
Q = 1.0*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0*Diagonal(I,m)

Qr = 1.0*Diagonal(I,n)
Qfr = 1.0*Diagonal(I,n)
Rr = 1.0*Diagonal(I,m)

x0 = [0; 0.; 0.]
xf = [1.0; 0.; 0.]
D = Diagonal((0.2^2)*ones(r))
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,constraint_tolerance=1.0e-3)

N = 101
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:rk3_implicit,dt)

u_max = 5.0
u_min = -5.0
function bnd_func(c,x,u)
    # for i = 1:m
    #     c[i] = u[i] - u_max
    #     c[i+m] = -u_min - u[i]
    # end

    c[1:m] = u .- u_max
    c[m .+ (1:m)] = -u_min .- u
    return nothing
end
bnd_con = Constraint{Inequality}(bnd_func,n,m,2m,:bound)

n_robust = 2*n^2 + n*r # number of robust parameters in state vector
n̄ = n + n_robust
m1 = m + n^2
idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),s=((n+n^2+n*r) .+ (1:n^2)),z=(1:n̄))

length(idx.z)
f(ẋ,z) = model.f(ẋ,z[1:n],z[n .+ (1:m)],zeros(eltype(z),r))
∇f(z) = ForwardDiff.jacobian(f,zeros(eltype(z),n),z)
∇f(x,u) = ∇f([x;u])

function K(z,u)
    x = z[idx.x]
    s = z[idx.s]
    P = reshape(s,n,n)*reshape(s,n,n)'
    Bc = ∇f(x,u)[:,n .+ (1:m)]
    R\(Bc'*P)
end

bnd_con_robust = robust_constraint(bnd_con,K,idx,n,m,n̄)

x = rand(n)
u = rand(m)
E = vec(3.0*Matrix(I,n,n))
H = vec(rand(n,r))
P = vec(5.0*Matrix(I,n,n))

z = [x;E;H;P]

p = length(bnd_con)
c = zeros(p)
C = zeros(p,n+m)
c_robust = zeros(bnd_con_robust.p)
C_robust = zeros(bnd_con_robust.p,n̄+m)

bnd_con.c(c,x,u)
bnd_con.∇c(C,x,u)
c
C

bnd_con_robust.c(c_robust,z,u)
bnd_con_robust.∇c(C_robust,z,u)
c_robust
C_robust

@test c_robust[1:length(bnd_con)] == c
@test C_robust[1:length(bnd_con),end-(m-1):end] == C[:,end-(m-1):end]

δu = get_δu(K,z,u,idx,1)

@test isapprox(c_robust[p .+ (1:m)],(u + δu) .- u_max)
@test isapprox(c_robust[(p+m) .+ (1:m)],-u_min .- (u + δu))

@test isapprox(c_robust[2*p .+ (1:m)],(u - δu) .- u_max)
@test isapprox(c_robust[(2*p+m) .+ (1:m)],-u_min .- (u - δu))

δu = get_δu(K,z,u,idx,2)
@test isapprox(c_robust[3*p .+ (1:m)],(u + δu) .- u_max)
@test isapprox(c_robust[(3*p+m) .+ (1:m)],-u_min .- (u + δu))

@test isapprox(c_robust[4p .+ (1:m)],(u - δu) .- u_max)
@test isapprox(c_robust[(4p+m) .+ (1:m)],-u_min .- (u - δu))

@test isapprox(c_robust[5*p .+ (1:m)],u .- u_max)
@test isapprox(c_robust[(5*p+m) .+ (1:m)],-u_min .- u)

@test all(C_robust[1:p,n .+ (1:n^2)] .== 0.)
@test C_robust[p .+ (1:p), end-(m-1):end] == C[:,end-(m-1):end]
C_robust[p .+ (1:p),n .+ (1:n^2)]


∇δu(K,z,u,idx,1)

function f1(y)
    x = y[idx.x]
    s = y[idx.s]
    u = y[length(idx.z)+1:end]
    P = reshape(s,n,n)*reshape(s,n,n)'
    Bc = ∇f(x,u)[:,n .+ (1:m)]
    _K = R\(Bc'*P)
    E = reshape(y[idx.e],n,n)
    tmp = _K*E*_K'
    # ee = eigen(tmp)
    # b = Diagonal(sqrt.(ee.values))*ee.vectors'
    ee = svd(tmp)
    b = ee.U*Diagonal(sqrt.(ee.S))*ee.Vt
    b[:,1]
end


f1([z;u])
ForwardDiff.jacobian(f1,[z;u])
K_aug(y) = vec(K(y[idx.z],y[(length(idx.z)+1):end]))

ForwardDiff.jacobian(K_aug,[z;u])[:,1:n]#[:,(n+n^2+n*r+n^2) .+ (1:m)]



_K = K(z[idx.z],u)
E = reshape(z[idx.e],n,n)
tmp = _K*E*_K'
ee = svd(tmp)
b = ee.U*Diagonal(sqrt.(ee.S))*ee.Vt
b*b

@test isapprox(b*b, tmp)
