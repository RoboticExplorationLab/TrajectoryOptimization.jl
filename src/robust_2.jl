using Plots
# model
T = Float64
model = TrajectoryOptimization.Dynamics.doubleintegrator_model_uncertain
n = model.n; m = model.m; r = model.r

# costs
Q = 1.0*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0.0; 0.0]
xf = [1.0; 0.0]

D = [0.2^2]
E1 = (1e-6)*ones(n,n)
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-8)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-5)

N = 100
tf = 1.0
U0 = [rand(m) for k = 1:N-1]

prob = TrajectoryOptimization.Problem(model, TrajectoryOptimization.Objective(costfun,N), integration=:rk4, x0=x0, N=N, tf=tf)
dt = prob.dt
TrajectoryOptimization.initial_controls!(prob, U0)

solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
TrajectoryOptimization.solve!(prob, solver_ilqr)
@test norm(prob.X[N] - xf) < 1e-3
plot(prob.X)

K,P = tvlqr_dis(prob,Q,R,Qf)
P[end]
P_vec = [vec(P[k]) for k = 1:N]
plot(P_vec)

idx = (x=1:n,u=1:m,w=1:r)
z_idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),p=((n+n^2+n*r) .+ (1:n^2)))

Zc = zeros(n,n+m+r)
n̄ = n + n^2 + n*r + n^2

function robust_dynamics(ż,z,u,w)
    x = z[z_idx.x]
    E = reshape(z[z_idx.e],n,n)
    H = reshape(z[z_idx.h],n,r)
    P = reshape(z[z_idx.p],n,n)

    model.∇f(Zc,x,u,w)
    Ac = Zc[:,idx.x]
    Bc = Zc[:,n .+ idx.u]
    Gc = Zc[:,(n+m) .+ idx.w]

    Kc = R\(Bc'*P)
    Acl = Ac - Bc*Kc

    model.f(view(ż,z_idx.x),x,u,w)
    ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
    ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
    ż[z_idx.p] = reshape((Ac'*P + P*Ac - P*Bc*Kc + Q),n^2)
end

robust_dynamics_dis = rk4_uncertain(robust_dynamics,dt)

Z = [zeros(n̄) for k = 1:N]
Z[1][z_idx.x] = x0
Z[1][z_idx.e] = vec(E1)
Z[1][z_idx.h] = vec(H1)
Z[1][z_idx.p] = vec(P[1])


for k = 1:N-1
    robust_dynamics_dis(Z[k+1],Z[k],prob.U[k],zeros(r))
end

X = [Z[k][z_idx.x] for k = 1:N]
E = [Z[k][z_idx.e] for k = 1:N]
H = [Z[k][z_idx.h] for k = 1:N]
P = [Z[k][z_idx.p] for k = 1:N]

plot(X)
plot(E)
plot(H)
plot(P)
plot!(P_vec,style=:dash,legend=:top)


##
function ric_dynamics!(ż,z,u,w)
    x = z[z_idx.x]
    E = reshape(z[z_idx.e],n,n)
    H = reshape(z[z_idx.h],n,r)
    P = reshape(z[z_idx.p],n,n)

    model.∇f(Zc,x,u,w)
    Ac = Zc[:,idx.x]
    Bc = Zc[:,n .+ idx.u]
    Gc = Zc[:,(n+m) .+ idx.w]

    Kc = R\(Bc'*P)
    Acl = Ac - Bc*Kc

    model.f(view(ż,z_idx.x),x,u,w)
    ż[z_idx.x] .*= 1.0
    ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
    ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
    ż[z_idx.p] = reshape(-1.0*(Ac'*P + P*Ac - P*Bc*Kc + Q),n^2)
end
model.∇f(Zc,prob.X[end],prob.U[end],zeros(r))
Zc
zz = zeros(n̄)
zz[z_idx.x] = xf
zz[z_idx.p] = vec(Qf)


function ric_ode(u,ww,t)
    uu = zero(u)
    ric_dynamics!(uu,u,U_interp(t),zeros(r))
    uu
end

zz = zeros(n̄)
zz[z_idx.x] = xf
zz[z_idx.p] = vec(Qf)
u0=zz
tspan = (tf,0.0)
pro = ODEProblem(ric_ode,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,RK4(),dt=dt)#,reltol=1e-8,abstol=1e-8)
p = plot()
for i = ((n+n^2+n*r) .+ (1:n^2))
    p = plot!(sol.t[end:-1:1],to_array(sol.u)[i,end:-1:1],legend=:top)
end
# for i = 1:n^2
#     p = plot!(range(0,stop=tf,length=N),to_array(P_vec)[i,:],style=:dash,legend=:top)
# end
display(p)

zz[z_idx.x] = x0
zz[z_idx.p] = sol.u[end][z_idx.p]
u0=zz
tspan = (0.0,tf)
pro = ODEProblem(ric_ode,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,RK4(),dt=dt)#,reltol=1e-8,abstol=1e-8)
p = plot()
for i = ((n+n^2+n*r) .+ (1:n^2))
    p = plot!(sol.t[end:-1:1],to_array(sol.u)[i,end:-1:1])
end
for i = 1:n^2
    p = plot!(range(0,stop=tf,length=N),to_array(P_vec)[i,:],style=:dash,legend=:top)
end
display(p)
sol.u
