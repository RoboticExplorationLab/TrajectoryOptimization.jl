using Plots, OrdinaryDiffEq
# model
T = Float64
model = TrajectoryOptimization.Dynamics.doubleintegrator_model_uncertain
n = model.n; m = model.m; r = model.r

# costs
Q = 1.0e-1*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0.0; 0.0]
xf = [pi; 0.0]

D = 0.2^2 * Matrix(I,r,r)
E1 = (1e-6)*ones(n,n)
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-8)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-5)

N = 100
tf = 3.0
U0 = [ones(m) for k = 1:N-1]

prob = TrajectoryOptimization.Problem(model, TrajectoryOptimization.Objective(costfun,N), integration=:rk4, x0=x0, N=N, tf=tf)
dt = prob.dt
TrajectoryOptimization.initial_controls!(prob, U0)

rollout!(prob)
# solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
# TrajectoryOptimization.solve!(prob, solver_ilqr)
# @test norm(prob.X[N] - xf) < 1e-3
plot(prob.X)

K,P = tvlqr_dis(prob,Q,R,Qf)
isposdef(0.5*(P[1] + P[1]'))
isposdef(P[1])
P_vec = [vec(P[k]) for k = 1:N]
plot(P_vec,linetype=:steppost,legend=:top)

idx = (x=1:n,u=1:m,w=1:r)
z_idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),p=((n+n^2+n*r) .+ (1:n^2)))

Zc = zeros(n,n+m+r)
n̄ = n + n^2 + n*r + n^2

function robust_dynamics(ż,z,u,w)
    x = z[z_idx.x]
    E = reshape(z[z_idx.e],n,n)
    H = reshape(z[z_idx.h],n,r)
    P = reshape(z[z_idx.p],n,n)
    P = 0.5*(P + P')

    model.∇f(Zc,x,u,w)
    Ac = Zc[:,idx.x]
    Bc = Zc[:,n .+ idx.u]
    Gc = Zc[:,(n+m) .+ idx.w]

    Kc = R\(Bc'*P)
    Acl = Ac - Bc*Kc

    model.f(view(ż,z_idx.x),x,u,w)
    # ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
    # ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
    ż[z_idx.p] = reshape(-1.0*(Ac'*P + P*Ac - P*Bc*Kc + Q),n^2)
end

NN = 10*N
Z = [zeros(n̄) for k = 1:NN]
Z[1][z_idx.x] = x0
Z[1][z_idx.e] = vec(E1)
Z[1][z_idx.h] = vec(H1)
Z[1][z_idx.p] = vec(P[1])

_dt = tf/(NN-1)
_t = [0.0]
# U_rev = prob.U[end:-1:1]
robust_dynamics_dis = rk4_uncertain(robust_dynamics,_dt)

for k = 1:NN-1
    robust_dynamics_dis(Z[k+1],Z[k],U_interp(_t[1]),zeros(r),_dt)
    _t[1] += _dt
    # println(_t[1])
    # if _t[1] < 0.0
    #     println("negative time")
    #     break
    # end
end

X = [Z[k][z_idx.x] for k = 1:NN]
E = [Z[k][z_idx.e] for k = 1:N]
H = [Z[k][z_idx.h] for k = 1:N]
P = [Z[k][z_idx.p] for k = 1:NN]

plot(X)
# plot!(prob.X)
plot(E)
plot(H)
plot(P)
plot(P_vec)

_t_ran = range(0,stop=_dt*N,length=N)
t_ran = range(0,stop=tf,length=N)
p11 = [P_vec[k][1] for k = 1:N]
_p11 = [P[k][1] for k = 1:N]
plot(_t_ran,_p11)

plot!(t_ran,p11,style=:dash,legend=:top)

##
function ric_dynamics!(ż,z,u,w)
    x = z[z_idx.x]
    E = reshape(z[z_idx.e],n,n)
    H = reshape(z[z_idx.h],n,r)
    P = reshape(z[z_idx.p],n,n)
    P = 0.5*(P + P')

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

zr = rand(n̄)
żr1 = zeros(n̄)
żr2 = zeros(n̄)

robust_dynamics(żr1,zr,prob.U[10],zeros(r))
ric_dynamics!(żr2,zr,prob.U[10],zeros(r))
żr1














żr2


model.∇f(Zc,prob.X[end],prob.U[end],zeros(r))
Zc
zz = zeros(n̄)
zz[z_idx.x] = xf
zz[z_idx.p] = vec(Qf)


X_interp = gen_cubic_interp(prob.X,prob.dt)
U_interp = gen_zoh_interp([prob.U...,prob.U[end]],prob.dt)
function ric_ode(u,ww,t)
    uu = zero(u)
    robust_dynamics(uu,u,U_interp(t),zeros(r))
    uu
end


zz = zeros(n̄)
zz[z_idx.x] = xf
zz[z_idx.p] = vec(Qf)
u0=zz
tspan = (tf,0.0)
pro = ODEProblem(ric_ode,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,Tsit5(),dt=dt)#,reltol=1e-8,abstol=1e-8)
p = plot()
for i = ((n+n^2+n*r) .+ (1:n^2))
    p = plot!(sol.t[end:-1:1],to_array(sol.u)[i,end:-1:1],legend=:top)
end
# for i = 1:n^2
#     p = plot!(range(0,stop=tf,length=N),to_array(P_vec)[i,:],style=:dash,legend=:top)
# end
display(p)

zz[z_idx.x] = x0
_P1 = reshape(sol.u[end][z_idx.p],n,n)
zz[z_idx.p] = 0.5*(_P1 + _P1')
u0=zz
tspan = (0.0,tf)
pro = ODEProblem(ric_ode,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,Tsit5(),dt=dt,reltol=1e-8,abstol=1e-8)
plot(sol.t)
p = plot()
for i = 1:((n+n^2+n*r) .+ (1:n^2))
    p = plot!(sol.t[end:-1:1],to_array(sol.u)[i,end:-1:1],legend=:top)
end
# for i = 1:n^2
#     p = plot!(range(0,stop=tf,length=N),to_array(P_vec)[i,:],style=:dash,legend=:top)
# end
display(p)
sol.u
