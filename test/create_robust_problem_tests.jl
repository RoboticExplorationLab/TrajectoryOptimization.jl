using Test

model = Dynamics.pendulum_model_uncertain
n = model.n; m = model.m; r = model.r

T = Float64

# costs
Q = 1.0*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0*Diagonal(I,m)

Qr = copy(Q)
Qfr = copy(Qf)
Rr = copy(R)

x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)

D = (0.2^2)*ones(r)
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,r)

costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)

verbose = true
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-3)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-2)

N = 501
tf = 1.0
dt = tf/(N-1)
U0 = [ones(m) for k = 1:N-1]

model_d = discretize_model(model,:midpoint_implicit,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)

model_d.f(zeros(n),rand(n),rand(m),rand(r),dt)
model_d.∇f(zeros(n,n+m+r+1),rand(n),rand(m),rand(r),dt)
F = PartedArray(zeros(T,n,length(prob.model)),create_partition2(prob.model))
jacobian!(F,prob.model,zeros(n),zeros(m),1.0)

initial_controls!(prob,U0)
rollout!(prob)
plot(prob.X)
Kd,Pd = tvlqr_dis(prob,Q,R,Qf)
Kc,Pc = tvlqr_con(prob,Q,R,Qf,xf)
#
Pd_vec = [vec(Pd[k]) for k = 1:N]
Pc_vec = [vec(Pc[k]) for k = 1:N]

@test norm(Pd_vec[1] - Pc_vec[1]) < 0.1
S1 = vec(sqrt(Pc[1]))
# plot(Pd_vec,label="",linetype=:steppost)
# plot!(Pc_vec,label="")

prob_robust = robust_problem(prob,E1,H1,D,Q,R,Qf,Q,R,Qf,xf)

n̄ = n + n^2 + n*r + n^2
m1 = m + n^2
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

initial_controls!(prob_robust,U0)

rollout!(prob_robust)
@test prob_robust.X[1][1:n] == x0
@test prob_robust.X[1][end-n^2+1:end] == S1
@test prob_robust.U[1][1:m] == U0[1]
@test prob_robust.X[end][1:n] == prob.X[end]

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

# jacobian!(prob_robust,ilqr_solver)
#
# backwardpass!(prob_robust,ilqr_solver)
# forwardpass!(prob_robust,ilqr_solver,[0.;0.],1e8)
#
# cost_expansion!(ilqr_solver.Q,prob_robust.obj,prob_robust.X,prob_robust.U)
al_solver = AbstractSolver(prob_robust,opts_al)
prob_al = AugmentedLagrangianProblem(prob_robust,al_solver)
solve!(prob_robust,opts_al)
