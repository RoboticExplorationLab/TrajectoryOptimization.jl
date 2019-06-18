using Test

model = Dynamics.doubleintegrator_model_uncertain
n = model.n; m = model.m; r = model.r

T = Float64

# costs
Q = 1.0*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0*Diagonal(I,m)

Qr = 1.0*Diagonal(I,n)
Qfr = 1.0*Diagonal(I,n)
Rr = 1.0*Diagonal(I,m)

x0 = [0; 0.]
xf = [1.0; 0]
D = Diagonal((0.2^2)*ones(r))
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,live_plotting=:state)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,penalty_scaling=10.0,constraint_tolerance=1.0e-4)

N = 101
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:rk3_implicit,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)

# ilqr_solver_i = AbstractSolver(prob,opts_ilqr)
# jacobian!(prob,ilqr_solver_i)
# cost_expansion!(prob,ilqr_solver_i)
# ΔV = backwardpass!(prob,ilqr_solver_i)
# forwardpass!(prob,ilqr_solver_i,ΔV,J)
#
# model_d.f(zeros(n),rand(n),rand(m),rand(r),dt)
# model_d.∇f(zeros(n,n+m+r+1),rand(n),rand(m),rand(r),dt)
# F = PartedArray(zeros(T,n,length(prob.model)),create_partition2(prob.model))
# jacobian!(F,prob.model,zeros(n),zeros(m),1.0)

initial_controls!(prob,U0)
# rollout!(prob)
solve!(prob,opts_ilqr)
plot(prob.X)
plot(prob.U)
Kd,Pd = tvlqr_dis(prob,Qr,Rr,Qfr)
Kc,Sc = tvlqr_sqrt_con_rk3_uncertain(prob,Qr,Rr,Qfr,xf)
#
Pd_vec = [vec(Pd[k]) for k = 1:N]
Pc_vec = [vec(reshape(Sc[k],n,n)*reshape(Sc[k],n,n)') for k = 1:N]
S1 = copy(Sc[1])

# @test norm(Pd_vec[1] - Pc_vec[1]) < 0.1
plot(Pd_vec,label="",linetype=:steppost)
plot!(Pc_vec,label="")

#create robust problem
prob_robust = robust_problem(prob,E1,H1,D,Qr,Rr,Qfr,Q,R,Qf,xf)

n̄ = n + n^2 + n*r + n^2
m1 = m + n^2

@test length(prob_robust.X[1]) == n̄
@test prob_robust.x0[(n+n^2+n*r) .+ (1:n^2)] == S1
@test size(prob_robust.U[1]) == (m1,)
@test prob_robust.model.n == n̄
@test prob_robust.model.m == m
@test prob_robust.model.r == r

k = 1
@test size(prob_robust.obj.obj.cost[k].Q) == (n̄,n̄)
@test size(prob_robust.obj.obj.cost[k].R) == (m1,m1)
@test size(prob_robust.obj.obj.cost[k].q) == (n̄,)
@test size(prob_robust.obj.obj.cost[k].r) == (m1,)
@test size(prob_robust.obj.obj.cost[k].H) == (m1,n̄)
for k = 2:N-1
    @test size(prob_robust.obj.obj.cost[k].Q) == (n̄,n̄)
    @test size(prob_robust.obj.obj.cost[k].R) == (m,m)
    @test size(prob_robust.obj.obj.cost[k].q) == (n̄,)
    @test size(prob_robust.obj.obj.cost[k].r) == (m,)
    @test size(prob_robust.obj.obj.cost[k].H) == (m,n̄)
end
k = N
@test size(prob_robust.obj.obj.cost[k].Qf) == (n̄,n̄)
@test size(prob_robust.obj.obj.cost[k].qf) == (n̄,)

@test prob_robust.obj.robust_cost.n == n
@test prob_robust.obj.robust_cost.m == m
@test prob_robust.obj.robust_cost.r == r

@test size(prob_robust.obj.robust_cost.K(rand(n̄),rand(m))) == (m,n)

c = zeros(n^2)
prob_robust.constraints[1][1].c(c,ones(n̄),0.5*ones(m1))
@test c == -0.5*ones(n^2)

C = zeros(n^2,n̄+m1)
prob_robust.constraints[1][1].∇c(C,rand(n̄+m1),rand(m1))
@test C[:,(n+n^2+n*r) .+ (1:n^2)] == -1.0*Matrix(I,n^2,n^2)
@test C[:,(n̄+m) .+ (1:n^2)] == 1.0*Matrix(I,n^2,n^2)

initial_controls!(prob_robust,prob.U)

copyto!(prob_robust.X[1],prob_robust.x0)

for k = 1:N-1
    println(k)
    prob_robust.model.f(prob_robust.X[k+1],prob_robust.X[k],prob_robust.U[k],zeros(r),dt)
end
prob_robust.X
plot(prob_robust.X,legend=:left)

idx = prob_robust.obj.robust_cost.idx
pp = [vec(reshape(prob_robust.X[k][idx.s],n,n)*reshape(prob_robust.X[k][idx.s],n,n)') for k = 1:N]
plot(pp)

@test prob_robust.X[1][1:n] == x0
@test prob_robust.X[1][end-n^2+1:end] == S1
@test prob_robust.U[1][1:m] == prob.U[1]
@test isapprox(prob_robust.X[end][1:n],prob.X[end])

ilqr_solver = AbstractSolver(prob_robust,iLQRSolverOptions())
@test size(ilqr_solver.Ū[1]) == (m1,)
@test size(ilqr_solver.K[1]) == (m1,n̄)
@test size(ilqr_solver.d[1]) == (m1,)
@test size(ilqr_solver.∇F[1]) == (n̄,n̄+m1+r+1)
@test size(ilqr_solver.Q[1].uu) == (m1,m1)
@test size(ilqr_solver.Q[1].ux) == (m1,n̄)
@test size(ilqr_solver.Q[1].u) == (m1,)
@test size(ilqr_solver.Q[1].xx) == (n̄,n̄)
@test size(ilqr_solver.Q[1].x) == (n̄,)

# rollout!(prob_robust)
# J = cost(prob_robust)
# xx = [prob_robust.X[k][idx.x] for k = 1:N]
# ee = [prob_robust.X[k][idx.e] for k = 1:N]
# ss = [prob_robust.X[k][idx.s] for k = 1:N]
# pp = [vec(reshape(prob_robust.X[k][idx.s],n,n)*reshape(prob_robust.X[k][idx.s],n,n)') for k = 1:N]
# plot(xx)
# plot!(prob.X)
# plot(ee)
# plot(ss)
# plot(pp)

prob_robust = robust_problem(prob,E1,H1,D,Qr,Rr,Qfr,Q,R,Qf,xf)

rollout!(prob_robust)

# jacobian!(prob_robust,ilqr_solver)
# cost_expansion!(prob_robust,ilqr_solver)
# ΔV = backwardpass!(prob_robust,ilqr_solver)
#
# ilqr_solver
# rollout!(prob_robust,ilqr_solver,0.01)
#
# plot(prob_robust.X)
# plot!(ilqr_solver.X̄)
# _J = cost(prob)
# forwardpass!(prob_robust,ilqr_solver,ΔV,J)
# prob_robust.X

al_solver = AbstractSolver(prob_robust,opts_al)

solve!(prob_robust,al_solver)

# max_violation(al_solver)
#
# prob_al
#
# plot(prob_al.X)
#
#
# plot(prob_robust.X)
# plot!(prob_al.X)
# solve!(prob_robust,opts_al)
