# model
T = Float64
model = TrajectoryOptimization.Dynamics.pendulum_model_uncertain
n = model.n; m = model.m; r = model.r

# costs
Q = 1.0e-1*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = false
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-8)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-5)

N = 201
dt = 0.005
U0 = [ones(m) for k = 1:N-1]

prob = TrajectoryOptimization.Problem(model, TrajectoryOptimization.Objective(costfun,N), integration=:rk4, x0=x0, N=N, dt=dt)
TrajectoryOptimization.initial_controls!(prob, U0)

plot(prob.X)
solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
TrajectoryOptimization.solve!(prob, solver_ilqr)

U_sol = copy(prob.U)
@test norm(prob.X[N] - xf) < 1e-3

# multi-cost
m_cost = add_costs(costfun,costfun)
@test m_cost isa MultiCost
@test m_cost isa CostFunction
@test length(m_cost.cost) == 2
add_cost!(m_cost,costfun)
@test length(m_cost.cost) == 3
prob_new = update_problem(prob,obj=Objective(m_cost,N))

TrajectoryOptimization.initial_controls!(prob_new, U0)
solver_ilqr = TrajectoryOptimization.iLQRSolver(prob_new, opts_ilqr)
TrajectoryOptimization.solve!(prob_new, solver_ilqr)
@test norm(prob.X[N] - xf) < 1e-3

# Multi-objective
m_obj = add_objectives(prob.obj,prob.obj)
@test m_obj isa MultiObjective
@test m_obj isa AbstractObjective
@test length(m_obj.obj) == 2
add_objective!(m_obj,prob.obj)
@test length(m_obj.obj) == 3

prob = TrajectoryOptimization.Problem(model, m_obj, integration=:rk4, x0=x0, N=N, dt=dt)
TrajectoryOptimization.initial_controls!(prob, U0)
solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
TrajectoryOptimization.solve!(prob, solver_ilqr)
@test norm(prob.X[N] - xf) < 1e-3

## Robust cost
nx,nu,nw = model.n,model.m,model.r;

x0 = [0.;0.]

# cost functions
R = Rr = [0.1]
Q = Qr = [10. 0.; 0. 1.]
Qf = Qfr = [100. 0.; 0. 100.]

# uncertainty
D = [0.2^2]
E1 = (0.0)*ones(nx,nx)
H1 = zeros(nx,nw)

# Robust Objective
obj = Objective(costfun,N)
# robust_costfun = RobustCost(prob.model.f,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
# robust_obj = RobustObjective(obj,robust_costfun)

x0 = zeros(n)
prob = TrajectoryOptimization.Problem(model, obj, integration=:rk4, x0=x0, N=N, dt=dt)
TrajectoryOptimization.initial_controls!(prob, U_sol)
rollout!(prob)
plot(prob.X)

fc_aug = f_augmented_uncertain!(model.f,n,m,r) #TODO make sure existing ∇f is ForwardDiff-able, then use that
∇fc(z) = ForwardDiff.jacobian(fc_aug,zeros(eltype(z),n),z)

fd_aug = f_augmented_uncertain!(prob.model.f,n,m,r) #TODO make sure existing ∇f is ForwardDiff-able, then use that
∇fd(z) = ForwardDiff.jacobian(fd_aug,zeros(eltype(z),n),z)

function back_dyn(ẏ,y,u)
    Dynamics.pendulum_model_uncertain_backward.f(view(ẏ,1:n),y[1:n],u,zeros(r))
    model.f(view(ẏ,1:n),y[1:n],u,zeros(r))

    Fc = ∇fc([y[1:n];u;zeros(r)])
    Ac = Fc[:,1:n]; Bc = Fc[:,n .+ (1:m)]; Gc = Fc[:,(n+m) .+ (1:r)]

    Fd = ∇fd([y[1:n];u;zeros(r);0.0])
    Ad = Fd[:,1:n]; Bd = Fd[:,n .+ (1:m)]
    Pc!(view(ẏ,n .+ (1:n^2)),reshape(y[n .+ (1:n^2)],n,n),Ac,Bc,Q,R)

end

back_dyn_d = rk4(back_dyn,dt)

Y = [zeros(n+n^2) for k = 1:N]
Y[1] .= [x0;reshape(P[1],n^2)]

xx = [zeros(n) for k = 1:N]
xx[1] = copy(xf)
u_rev = U_sol[end:-1:1]
for k = 1:N-1
    # prob.model.f(xx[k+1],xx[k],u_rev[k],zeros(r),prob.dt)
    back_dyn_d(Y[k+1],Y[k],U_sol[k],dt)
end
plot(xx)
xx[end]
prob.X[1]

Y[end]

P1 = Y[1][n .+ (1:n^2)]





Y = [zeros(n+n^2) for k = 1:N]
# Y[end] .= [Xf;reshape(Qf,n^2)]
Y[1] .= [x0;P1]


for k = 1:N-1
    # back_dyn_d(Y[k],Y[k+1],prob.U[k],-dt)
    back_dyn_d(Y[k+1],Y[k],prob.U[k],dt)

end

Y[end]

K,P = tvlqr(prob,Q,R,Qf)







# Xf = prob.X[end]
# Pf = reshape(Qf,n^2)
#
# Yf = [Xf;Pf]
#
# fc_aug = f_augmented_uncertain!(model.f,n,m,r) #TODO make sure existing ∇f is ForwardDiff-able, then use that
# ∇fc(z) = ForwardDiff.jacobian(fc_aug,zeros(eltype(z),n),z)
# ZZ = zeros(n,n+m+r+1)
# prob.model.∇f(ZZ,rand(n),prob.X[1],prob.U[1],zeros(r),dt)
# F = ∇fc([Xf;u_rev[1];zeros(r)])
# A = ZZ[:,1:n]; B = ZZ[:,n .+ (1:m)]
#
# function stacked(ẏ,y,u)
#     model.f(view(ẏ,1:n),y[1:n],u,zeros(r))
#     prob.model.∇f(ZZ,rand(n),y[1:n],u,zeros(r),-dt)
#     A = ZZ[:,1:n]; B = ZZ[:,n .+ (1:m)]
#     P = reshape(y[n .+ (1:n^2)],n,n)
#     ẏ[n .+ (1:n^2)] .= reshape(A'*P + P*A - P*B*(R\(B'*P)) + Q,n^2)
#
#     return nothing
# end
#
# stacked(rand(n+n^2),rand(n+n^2),rand(m))
#
# stacked_d = rk4(stacked,dt)
#
# y = [zeros(n+n^2) for k = 1:N]
# u_rev = prob.U[end:-1:1]
# y[1] .= Yf
# # y[1] .= [x0;reshape(P[1],n^2)]
# for k = 1:N-1
#     # stacked_d(y[k+1],y[k],prob.U[k],dt)
#     stacked_d(y[k+1],y[k],u_rev[k],-dt)
# end
# y[end]
#
# P = reshape(Pf,n,n)
# FF = ∇fc([prob.X[1];u_rev[1];zeros(r)])
# A = FF[:,1:n]
# B = FF[:,n .+ (1:m)]
# -1*(A'*P + P*A -P*B*(R\(B'*P)) + Q)
# y[end]
#
# P*B*(R\(B'*P))
# B'*P
#
#
#
#
#
# X_for = [zeros(n) for k = 1:N]
# X_back = [zeros(n) for k = 1:N]
# X_for[1] .= copy(x0)
#
#
# K, P = tvlqr(prob,Q,R,Qf)
# P[1]
# # robust_model(prob.model,E1,H1,D)
# # robust_model(model,E1,H1,D)
# fc_aug = f_augmented_uncertain!(model.f,n,m,r) #TODO make sure existing ∇f is ForwardDiff-able, then use that
# ∇fc(z) = ForwardDiff.jacobian(fc_aug,zeros(eltype(z),n),z)
# Kc(∇fc([prob.X[1];prob.U[1];zeros(r)])[:,n .+ (1:m)],P[1],R)
#
# prob.U[1]
#
# Zb = zeros(n,n+m+r+1)
#
# prob.model.∇f(Zb,prob.X[1],prob.U[1],zeros(r),dt)
# Kc(Zb[:,n .+ (1:m)],P[1],R)
#
# idx = (x = 1:n, u = 1:m, w = 1:r, e = n .+ (1:n^2), h = (n+n^2) .+ (1:n*r), p = (n+n^2+n*r) .+ (1:n^2))
#
# robust_cost = RobustCost(Q,R,Qf,Qr,Rr,Qfr,∇fc,n,m,r,idx)
#
# prob_robust = robust_problem(prob,E1,H1,D,Q,R,Qf,Qr,Rr,Qfr)
#
# rollout!(prob_robust)
#
# prob_robust
#
#
# n̄ = nx*(1 + 2*nx + nw)
# x̄ = rand(n̄)
# u = rand(nu)
# w = rand(nw)
# fc! = prob.model.info[:fc]
# fc_aug = f_augmented_uncertain!(fc!,nx,nu,nw)
# z̄ = [x̄[1:n];u;w]
# ∇fc(z) = ForwardDiff.jacobian(fc_aug,zeros(eltype(z),nx),z)
#
#
#
#
# idx = (x = 1:n, u = 1:m, w = 1:r, e = n .+ (1:n^2), h = (n+n^2) .+ (1:n*r), p = (n+n^2+n*r) .+ (1:n^2))
# function f_robust(ẏ::AbstractVector{T},y::AbstractVector{T},u::AbstractVector{T},w::AbstractVector{T}) where T
#     "
#         y = [x;E;H;P]
#     "
#     x = y[idx.x]
#
#     # E = reshape(y[idx.e],nx,nx)
#     # H = reshape(y[idx.h],nx,nw)
#     P = reshape(y[idx.p],nx,nx)
#
#     z = [x;u;w]
#     F = ∇fc(z)
#     A = F[:,idx.x]; B = F[:,nx .+ idx.u]; G = F[:,(nx+nu) .+ idx.w]
#     K = Kc(B,P,R)
#     Acl = (A - B*K)
#
#     # nominal dynamics
#     fc!(view(ẏ,1:n),x,u,w)
#     # disturbances
#     Ec!(view(ẏ,idx.e),E1,H1,Acl,G)
#     Hc!(view(ẏ,idx.h),H1,Acl,G,D)
#     # cost-to-go
#     Pc!(view(ẏ,idx.p),P,A,B,Q,R)
# end
#
# Kc(rand(nx,nu),rand(nx,nx),rand(nu,nu))
#
# y = rand(n̄)
# u = rand(nu)
# w = rand(nw)
#
# ẏ = zeros(n̄)
#
# f_robust(ẏ,y,u,w)
#
# ŷ = [y;u;w]
#
# f_robust_aug = f_augmented_uncertain!(f_robust,n̄,nu,nw)
#
# ff(ŷ) = ForwardDiff.jacobian(f_robust_aug,zeros(eltype(ŷ),n̄),ŷ)
#
#
#
# robust_model(model,E1,H1,D)
#
#
#
#
#
# prob_robust = robust_problem(prob,D,E1,Q,R,Qf,Qr,Rr,Qfr)
# nδ = 2*(nx^2 + nu^2)
# n̄ = n+nδ
# @test prob_robust.model.n == nx+nδ
# @test prob_robust.model.m == nu
# @test prob_robust.model.r == size(D,1)
# @test size(prob_robust.obj.obj[1].Q) == (n̄,n̄)
# @test size(prob_robust.obj.obj[1].q,1) == (n̄)
# @test size(prob_robust.obj.obj[N].Qf) == (n̄,n̄) .- 2*nu^2
# @test size(prob_robust.obj.obj[N].qf,1) == n̄ - 2*nu^2
# @test size(prob_robust.X[1],1) == n̄
# @test size(prob_robust.X[N],1) == n̄ - 2*nu^2
# @test size(prob_robust.x0,1) == n̄
#
# Z = zeros(n,n+m+r+1)
# _Z = zeros(n,n̄+m+r+1)
# ∇f = prob.model.∇f
# _∇f = update_jacobian(∇f,n,m,r)
#
# x = rand(n); u = rand(m); w = rand(r); dt = 1.0
#
# idx_x = 1:n
# idx_u = 1:m
# idx = [(idx_x)...,(n̄ .+ (idx_u))...,((n̄+m) .+ (1:r))...,(n̄+m+r+1)]
# ∇f(Z,rand(n),x,u,w,dt)
# _∇f(_Z,rand(n),x,u,w,dt)
# _Z
# @test Z == view(_Z,idx_x,idx)
# _Z = zeros(n,n̄+m+r+1)
# _∇f(_Z,x,u,w,dt)
# @test Z == view(_Z,idx_x,idx)
# @test Z == view(_∇f(x,u,w,dt),idx_x,idx)
#
# x0 = ones(n)
# δx = [0.1*rand(n) for i = 1:n]
# xw = Vector[]
#
# u0 = ones(m)
# δu = [0.1*rand(m) for i = 1:m]
# uw = Vector[]
#
# push!(xw,x0)
# for i = 1:n
#     push!(xw,x0 + δx[i])
#     push!(xw,x0 - δx[i])
# end
#
# for j = 1:m
#     push!(uw,u0 + δu[j])
#     push!(uw,u0 - δu[j])
# end
# push!(uw,u0)
#
#
# xx = vcat(xw...,uw[1:2m]...)
# xxt = vcat(xw...)
# uu = uw[end]
#
# bnd = TrajectoryOptimization.bound_constraint(n, m, x_min=-1.0, x_max=1.0,u_min=-1.0, u_max=1.0,trim=true)
# goal = TrajectoryOptimization.goal_constraint(xf)
# bnd_robust = robust_constraint(bnd,n,m)
# goal_robust = robust_constraint(goal,n)
# cc = zeros(bnd_robust.p)
# CC = zeros(bnd_robust.p,n+2*(n^2+m^2)+m)
# cct = zeros(goal_robust.p)
# CCt = zeros(goal_robust.p,n+2*(n^2))
# bnd_robust.c(cc,xx,uu)
# bnd_robust.∇c(CC,xx,uu)
# cc
# CC
#
# goal_robust.c(cct,xxt)
# goal_robust.∇c(CCt,xxt)
# cct
# CCt
#
# @test cc[1:n] == xw[1] - ones(n)
# @test CC[1:n,1:n] == Diagonal(ones(n))
# @test cc[n+1] == (uw[1] - ones(m))[1]
# @test CC[n+1,n + 2*n^2 + 1] == 1.0
# @test cc[(n+1) .+ (1:n)] == -ones(n) - xw[1]
# @test CC[(n+1) .+ (1:n),(1:n)] == Diagonal(-ones(n))
# @test cc[(2*n+1)+1] == (-ones(m) - uw[1])[1]
# @test CC[2n+2,n + 2*n^2 + 1] == -1.0
#
# @test cct[1:n] == xw[1] - xf
# @test cct[n .+ (1:n)] == xw[2] - xf
# @test CCt == Diagonal(ones(goal_robust.p))
#
# x0 = zeros(n)
# prob = TrajectoryOptimization.Problem(model, obj,constraints=ProblemConstraints([bnd,goal],N), integration=:rk4, x0=x0, N=N, dt=dt)
# prob_robust = robust_problem(prob,D,E1,Q,R,Qf,Qr,Rr,Qfr)
# prob_robust.X[N]
# TrajectoryOptimization.initial_controls!(prob_robust, U0)
# rollout!(prob_robust)
# plot(prob_robust.X[1:end-1])
# # plot(prob_robust.X[1:end-1])
# cost(prob_robust)
# prob_robust.x0
# prob_robust.obj.robust_cost.ℓw(prob_robust.X,prob_robust.U)
# cost(prob_robust.obj.obj,prob_robust.X,prob_robust.U)
#
# prob_robust.X[1]
#
# # prob.obj.robust_cost.∇²ℓw
# # Qxx, Qux, Quu = unpack_cost_hessian(prob.obj.robust_cost.∇²ℓw,nx,nu,N)
# #
# # solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
# #
# # cost_expansion!(prob,solver_ilqr)
# # cost_expansion!(solver_ilqr.Q,prob.obj.robust_cost,prob.X,prob.U)
# # solver_ilqr.Q
#
# # Qxx, Qux, Quu = _unpack_robust_cost_hessian(c.∇²ℓw,nx,nu,N)
# # Qx, Qu = unpack(c.∇ℓw,nx,nu,N)
# #
# # typeof(prob.X[1])
# # typeof(c.∇ℓw)
# # typeof(c.∇²ℓw)
# #
# # c.∇²ℓw[1:2,1:2]
#
# function test1(x)
#     return x[1]+x[2], [x[3];x[3]]
# end
#
# ZZ = [zeros(ForwardDiff.Dual,2) for k = 1:3]
# ZZ = [zeros(2) for k = 1:3]
#
# function wrap_test1(x)
#     y,z = test1(x)
#     convert.(eltype(x),z)
#     y
# end
# x0 = rand(3)
# x0[1] + x0[2] == wrap_test1(x0)
#
# ForwardDiff.gradient(wrap_test1,x0)
# ZZ[1][1].value
# isapprox(ZZ[1][1],x0[3:3])
