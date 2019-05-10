using Interpolations, OrdinaryDiffEq

T = Float64
model = TrajectoryOptimization.Dynamics.doubleintegrator_model_uncertain
n = model.n; m = model.m; r = model.r

# costs
Q = 1.0*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.]#; 0.]
xf = [1.; 0.]#; 0.]

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

U_ = [prob.U..., prob.U[end]]
K,P = tvlqr_dis(prob,Q,R,Qf)

K_vec = [reshape(K[k],n*m) for k = 1:N-1]
P_vec = [reshape(P[k],n^2) for k = 1:N]
plot(K_vec)
plot(P_vec)

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

# x interp
# plot(range(0,stop=total_time(prob),length=N),to_array(prob.X)[1,:])
# plot!(range(0,stop=total_time(prob),length=N),to_array(prob.X)[2,:])
#
# t_ = range(0,stop=total_time(prob),length=2*N)
# scatter!(t_,to_array(X_interp.(t_))[1,:])
# scatter!(t_,to_array(X_interp.(t_))[2,:])
#
# # u interp
# t_ = range(0,stop=total_time(prob)-prob.dt,length=N-1)
# plot(t_,to_array(prob.U)[1,:],linetype=:stairs)
#
# t_ = range(0,stop=total_time(prob),length=N)
# scatter!(t_,to_array(U_interp.(t_))[1,:])

Z = zeros(n,n+m+r)

function Ac(t)
    Z = zeros(n,n+m+r)
    x = X_interp(t)
    u = U_interp(t)
    model.∇f(Z,x,u,zeros(r))
    # model.∇f(Z,x,u)

    Z[:,1:n]
end

function Bc(t)
    Z = zeros(n,n+m+r)
    x = X_interp(t)
    u = U_interp(t)
    model.∇f(Z,x,u,zeros(r))
    # model.∇f(Z,x,u)

    Z[:,n .+ (1:m)]
end

function Gc(t)
    Z = zeros(n,n+m+r)
    x = X_interp(t)
    u = U_interp(t)
    model.∇f(Z,x,u,zeros(r))
    # model.∇f(Z,x,u)

    Z[:,(n+m) .+ (1:r)]
end

Zd = zeros(n,n+m+r+1)
prob.model.∇f(Zd,rand(n),rand(m),rand(r),dt)

function riccati(p,w,t)
    n = convert(Int64,sqrt(length(p)))
    P = reshape(p,n,n)
    println(t)
    println(Ac(t))
    println(Bc(t))
    -1.0*reshape(Ac(t)'*P + P*Ac(t) - P*Bc(t)*(R\(Bc(t)'*reshape(P,n,n))) + Q, n^2)
end

# function riccati2(p,w,t)
#     n = convert(Int64,sqrt(length(p)))
#     P = reshape(p,n,n)
#     reshape(Ac(t)'*P + P*Ac(t) - P*Bc(t)*(R\(Bc(t)'*reshape(P,n,n))) + Q, n^2)
# end

u0=vec(Qf)
tspan = (tf,0.0)
pro = ODEProblem(riccati,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,Midpoint(),dt=dt)#,reltol=1e-8,abstol=1e-8)
p = plot()
for i = 1:n^2
    p = plot!(sol.t[end:-1:1],to_array(sol.u)[i,end:-1:1])
end
for i = 1:n^2
    p = plot!(range(0,stop=tf,length=N),to_array(P_vec)[i,:],style=:dash,legend=:top)
end
display(p)

P1 = sol.u[end]

# u0=vec(round.(P1,digits=6))
u0 = vec(P1)
tspan = (0,tf)
pro = ODEProblem(riccati,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,Midpoint(),dt=dt)#,reltol=1e-6,abstol=1e-6)
p = plot()
for i = 1:n^2
    p = plot!(sol.t,to_array(sol.u)[i,:],legend=:none)
end
for i = 1:n^2
    p = plot!(range(0,stop=tf,length=N),to_array(P_vec)[i,:],style=:dash,legend=:topleft)
end
display(p)
sol.u[end]

function dyn(x,w,t)
    xx = zero(x)
    model.f(xx,x,U_interp(t),zeros(r))
    xx
end

u0=xf
tspan = (tf,0)
pro = ODEProblem(dyn,u0,tspan)
sol = OrdinaryDiffEq.solve(pro,Tsit5())#,reltol=1e-6,abstol=1e-6)
NN = length(sol.u)
x1 = [sol.u[k][1] for k = 1:NN]
x2 = [sol.u[k][2] for k = 1:NN]

plot(sol.t,x1)
plot!(sol.t,x2)

# idx = (x = 1:n, u = 1:m, w = 1:r, e = n .+ (1:n^2), h = (n+n^2) .+ (1:n*r), p = (n+n^2+n*r) .+ (1:n^2))
# function f_robust!(ẏ,y,t)
#
#     x = y[idx.x]
#     u = U_interp(t)
#     w = zeros(r)
#     P = y[idx.p]
#     E = reshape(y[idx.e],n,n)
#     H = reshape(y[idx.h],n,r)
#
#     A = Ac(t); B = Bc(t); G = Gc(t)
#     K = Kc(B,reshape(P,n,n),R)
#     Acl = A - B*K
#
#     # nominal dynamics
#     model.f(view(ẏ,1:n),x,u,w)
#
#     # # disturbances
#     Ec!(view(ẏ,idx.e),E,H,Acl,G)
#     Hc!(view(ẏ,idx.h),H,Acl,G,D)
#
#     # cost-to-go
#     riccati_con!(view(ẏ,idx.p),P,t)
# end
#
# function f_robust_ode(u,p,t)
#     uu = zero(u)
#     f_robust!(uu,u,t)
#     uu
# end
#
# u0=[x0;vec(E1);vec(H1);vec(P1)]
# tspan = (0,tf)
# pro = ODEProblem(f_robust_ode,u0,tspan)
# sol = OrdinaryDiffEq.solve(pro,AutoTsit5(Rosenbrock23()),dt=dt,reltol=1e-12,abstol=1e-12)
# sol.u[end]
# sol.u
# sol
# xx = [sol.u[k][idx.x] for k = 1:length(sol.u)]
# ee = [sol.u[k][idx.e] for k = 1:length(sol.u)]
# hh = [sol.u[k][idx.h] for k = 1:length(sol.u)]
# pp = [sol.u[k][idx.p] for k = 1:length(sol.u)]
#
# plot(xx)
# plot(ee)
# plot(hh)
# plot(pp)
#
# t_uniform = range(0,stop=tf,length=N)
# N_nu = length(sol.t)
# X_u = Vector[]
# push!(X_u,xx[1])
# for k = 2:2
#     idx = findall(t->t < t_uniform[k], sol.t)[end]
#     println(idx)
#     println(sol.t[idx], sol.t[idx+1])
#     sol.t[idx+1] - sol.t[idx+1]
# end
#
# length(sol.t)
# interpolate([[1.;1.], [2.;2.]],BSpline(Cubic(Line(OnGrid()))))(1.2)
#
# u0 = x0
# u0=prob.X[end]
#
# function _dynamics(u,p,t)
#     uu = zero(u)
#     model.f(uu,u,U_interp(t),zeros(r))
#     uu
# end
#
# function rev_dynamics(u,p,t)
#     uu = zero(u)
#     model.f(uu,u,U_interp(t),zeros(r))
#     return -1.0*uu
# end
# tspan = (0,tf)
# tspan = (tf,0)
# pro = ODEProblem(_dynamics,u0,tspan)
# sol = OrdinaryDiffEq.solve(pro,RK4())#,dt=dt)#,reltol=1e-6,abstol=1e-6)
#
# x1 = [sol.u[k][1] for k = 1:length(sol.u)]
# x2 = [sol.u[k][2] for k = 1:length(sol.u)]
#
# plot(sol.t,x1)
# plot!(sol.t,x2)
#
# u0 = sol.u[end]
#
# plot(sol.t[end:-1:1],x1[end:-1:1])
# plot!(sol.t[end:-1:1],x2[end:-1:1])
#
# _x1 = [prob.X[k][1] for k = 1:N]
# _x2 = [prob.X[k][2] for k = 1:N]
# _t = range(0,stop=tf,length=N)
#
#
# plot!(_t,_x1,style=:dash)
# plot!(_t,_x2,style=:dash,legend=:top)
#
# dyn_rk4 = rk4_uncertain(model.f,dt)
#
# function rev_dyn!(ẋ,x,u,w)
#     model.f(ẋ,x,u,w)
#     ẋ .*= -1.0
# end
#
# dyn_rk4_rev = rk4_uncertain(rev_dyn!,dt)
#
#
# xx = [zeros(n) for k = 1:N]
# xx[1] = x0
#
# tt = [0.]
# for k = 1:N-1
#     dyn_rk4(xx[k+1],xx[k],U_interp(tt[1]),zeros(r))
#     tt[1] += dt
# end
#
# plot(xx)
#
# xx = [zeros(n) for k = 1:N]
# xx[1] = xf
# u_rev = prob.U[end:-1:1]
# tt = [tf]
# UU = Vector{T}[]
# for k = 1:N-1
#     push!(UU,U_interp(tt[1]))
#     dyn_rk4(xx[k+1],xx[k],U_interp(tt[1]),zeros(r),-dt)
#     tt[1] -= dt
#     println(tt[1])
# end
# plot(xx[end:-1:1])
# plot!(prob.X)
# plot(UU)
# plot!(u_rev)
#
#
# xx = [zeros(n) for k = 1:N]
# xx[1] = xf
# tt = [tf]
# for k = 1:N-1
#     dyn_rk4_rev(xx[k+1],xx[k],u_rev[k],zeros(r),dt)
#     tt[1] -= dt
# end
#
# plot(xx)
#
# idx = (x=1:n,u=1:m,w=1:r)
# z_idx = (x=1:n,e=(n .+ (1:n^2)),h=((n+n^2) .+ (1:n*r)),p=((n+n^2+n*r) .+ (1:n^2)))
#
# Zc = zeros(n,n+m+r)
# n̄ = n + n^2 + n*r + n^2
#
# function robust_dynamics(ż,z,u,w)
#     x = z[z_idx.x]
#     E = reshape(z[z_idx.e],n,n)
#     H = reshape(z[z_idx.h],n,r)
#     P = reshape(z[z_idx.p],n,n)
#
#     model.∇f(Zc,x,u,w)
#     Ac = Zc[:,idx.x]
#     Bc = Zc[:,n .+ idx.u]
#     Gc = Zc[:,(n+m) .+ idx.w]
#
#     Kc = R\(Bc'*P)
#     Acl = Ac - Bc*Kc
#
#     model.f(view(ż,z_idx.x),x,u,w)
#     # ż[z_idx.e] = reshape(Acl*E + Gc*H' + E*Acl' + H*Gc',n^2)
#     # ż[z_idx.h] = reshape(Acl*H + Gc*D,n,r)
#     ż[z_idx.p] = reshape((Ac'*P + P*Ac - P*Bc*Kc + Q),n^2)
#     # ż[z_idx.p] = reshape((Ac'*reshape(P1,n,n) + reshape(P1,n,n)*Ac - reshape(P1,n,n)*Bc*Kc + Q),n^2)
# end
#
# robust_dynamics_dis = rk4_uncertain(robust_dynamics,dt)
# P1
# Z = [zeros(n̄) for k = 1:N]
# Z[1][z_idx.x] = x0
# Z[1][z_idx.e] = vec(E1)
# Z[1][z_idx.h] = vec(H1)
# Z[1][z_idx.p] = vec(P1)
#
# tt = [0.]
# for k = 1:N-1
#     robust_dynamics_dis(Z[k+1],Z[k],U_interp(tt[1]),zeros(r))
#     tt[1] += dt
# end
#
# X = [Z[k][z_idx.x] for k = 1:N]
# E = [Z[k][z_idx.e] for k = 1:N]
# H = [Z[k][z_idx.h] for k = 1:N]
# P = [Z[k][z_idx.p] for k = 1:N]
#
# plot(X)
# plot(E)
# plot(H)
# plot(P)
# plot!(P_vec[1:100],style=:dash)
#
#
#
#
