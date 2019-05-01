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
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-5)
opts_al = TrajectoryOptimization.AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,constraint_tolerance=1.0e-5)

N = 11
dt = 0.1
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
N = 11
nx,nu,nw = model.n,model.m,model.r;

dt = 0.1
x0 = [0.;0.]

# cost functions
R = Rr = [0.1]
Q = Qr = [10. 0.; 0. 1.]
Qf = Qfr = [100. 0.; 0. 100.]

# uncertainty
D = [0.2^2]
E1 = zeros(nx,nx)

# Robust Objective
obj = Objective(costfun,N)
robust_costfun = RobustCost(prob.model.f,D,E1,Q,R,Qf,Qr,Rr,Qfr,nx,nu,nw,N)
robust_obj = RobustObjective(obj,robust_costfun)

x0 = zeros(n)
prob = TrajectoryOptimization.Problem(model, obj, integration=:rk4, x0=x0, N=N, dt=dt)
prob_robust = robust_problem(prob,D,E1,Q,R,Qf,Qr,Rr,Qfr)
nδ = 2*(nx^2 + nu^2)
n̄ = n+nδ
@test prob_robust.model.n == nx+nδ
@test prob_robust.model.m == nu
@test prob_robust.model.r == size(D,1)
@test size(prob_robust.obj.obj[1].Q) == (n̄,n̄)
@test size(prob_robust.obj.obj[1].q,1) == (n̄)
@test size(prob_robust.obj.obj[N].Qf) == (n̄,n̄) .- 2*nu^2
@test size(prob_robust.obj.obj[N].qf,1) == n̄ - 2*nu^2
@test size(prob_robust.X[1],1) == n̄
@test size(prob_robust.X[N],1) == n̄ - 2*nu^2
@test size(prob_robust.x0,1) == n̄

Z = zeros(n,n+m+r+1)
_Z = zeros(n,n̄+m+r+1)
∇f = prob.model.∇f
_∇f = update_jacobian(∇f,n,m,r)

x = rand(n); u = rand(m); w = rand(r); dt = 1.0

idx_x = 1:n
idx_u = 1:m
idx = [(idx_x)...,(n̄ .+ (idx_u))...,((n̄+m) .+ (1:r))...,(n̄+m+r+1)]
∇f(Z,rand(n),x,u,w,dt)
_∇f(_Z,rand(n),x,u,w,dt)
_Z
@test Z == view(_Z,idx_x,idx)
_Z = zeros(n,n̄+m+r+1)
_∇f(_Z,x,u,w,dt)
@test Z == view(_Z,idx_x,idx)
@test Z == view(_∇f(x,u,w,dt),idx_x,idx)

x0 = ones(n)
δx = [0.1*rand(n) for i = 1:n]
xw = Vector[]

u0 = ones(m)
δu = [0.1*rand(m) for i = 1:m]
uw = Vector[]

push!(xw,x0)
for i = 1:n
    push!(xw,x0 + δx[i])
    push!(xw,x0 - δx[i])
end

for j = 1:m
    push!(uw,u0 + δu[j])
    push!(uw,u0 - δu[j])
end
push!(uw,u0)


xx = vcat(xw...,uw[1:2m]...)
xxt = vcat(xw...)
uu = uw[end]

bnd = TrajectoryOptimization.bound_constraint(n, m, x_min=-1.0, x_max=1.0,u_min=-1.0, u_max=1.0,trim=true)
goal = TrajectoryOptimization.goal_constraint(xf)
bnd_robust = robust_constraint(bnd,n,m)
goal_robust = robust_constraint(goal,n)
cc = zeros(bnd_robust.p)
CC = zeros(bnd_robust.p,n+2*(n^2+m^2)+m)
cct = zeros(goal_robust.p)
CCt = zeros(goal_robust.p,n+2*(n^2))
bnd_robust.c(cc,xx,uu)
bnd_robust.∇c(CC,xx,uu)
cc
CC

goal_robust.c(cct,xxt)
goal_robust.∇c(CCt,xxt)
cct
CCt

@test cc[1:n] == xw[1] - ones(n)
@test CC[1:n,1:n] == Diagonal(ones(n))
@test cc[n+1] == (uw[1] - ones(m))[1]
@test CC[n+1,n + 2*n^2 + 1] == 1.0
@test cc[(n+1) .+ (1:n)] == -ones(n) - xw[1]
@test CC[(n+1) .+ (1:n),(1:n)] == Diagonal(-ones(n))
@test cc[(2*n+1)+1] == (-ones(m) - uw[1])[1]
@test CC[2n+2,n + 2*n^2 + 1] == -1.0

@test cct[1:n] == xw[1] - xf
@test cct[n .+ (1:n)] == xw[2] - xf
@test CCt == Diagonal(ones(goal_robust.p))

x0 = zeros(n)
prob = TrajectoryOptimization.Problem(model, obj,constraints=ProblemConstraints([bnd,goal],N), integration=:rk4, x0=x0, N=N, dt=dt)
prob_robust = robust_problem(prob,D,E1,Q,R,Qf,Qr,Rr,Qfr)
prob_robust.X[N]
TrajectoryOptimization.initial_controls!(prob_robust, U0)
rollout!(prob_robust)
plot(prob_robust.X[1:end-1])
# plot(prob_robust.X[1:end-1])
cost(prob_robust)
prob_robust.x0
prob_robust.obj.robust_cost.ℓw(prob_robust.X,prob_robust.U)
cost(prob_robust.obj.obj,prob_robust.X,prob_robust.U)

prob_robust.X[1]

# prob.obj.robust_cost.∇²ℓw
# Qxx, Qux, Quu = unpack_cost_hessian(prob.obj.robust_cost.∇²ℓw,nx,nu,N)
#
# solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)
#
# cost_expansion!(prob,solver_ilqr)
# cost_expansion!(solver_ilqr.Q,prob.obj.robust_cost,prob.X,prob.U)
# solver_ilqr.Q

# Qxx, Qux, Quu = _unpack_robust_cost_hessian(c.∇²ℓw,nx,nu,N)
# Qx, Qu = unpack(c.∇ℓw,nx,nu,N)
#
# typeof(prob.X[1])
# typeof(c.∇ℓw)
# typeof(c.∇²ℓw)
#
# c.∇²ℓw[1:2,1:2]

function test1(x)
    return x[1]+x[2], [x[3];x[3]]
end

ZZ = [zeros(ForwardDiff.Dual,2) for k = 1:3]
ZZ = [zeros(2) for k = 1:3]

function wrap_test1(x)
    y,z = test1(x)
    convert.(eltype(x),z)
    y
end
x0 = rand(3)
x0[1] + x0[2] == wrap_test1(x0)

ForwardDiff.gradient(wrap_test1,x0)
ZZ[1][1].value
isapprox(ZZ[1][1],x0[3:3])
