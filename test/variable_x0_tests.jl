using Test

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
opts_ilqr = TrajectoryOptimization.iLQRSolverOptions{T}(verbose=verbose,cost_tolerance=1.0e-6)

N = 51
dt = 0.1
U0 = [rand(m) for k = 1:N-1]

model_d = discretize_model(model,:rk4,dt)
prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)
TrajectoryOptimization.initial_controls!(prob, U0)

# nominal rollout
rollout!(prob)

# variable x0
_n = 3
m1 = m + _n
n1 = n + _n
_y0 = rand(_n)

dyn_mod(ẋ,x,u,w) = let n=n,m=m
    model.f(view(ẋ,1:n),x[1:n],u[1:m],w)
end

ẋ1 = zeros(n1)
dyn_mod(ẋ1,rand(n1),rand(m1),rand(r))
@test all(ẋ1[n .+ (1:_n)] .== 0)
ẋ2 = zeros(n)
_ẋ2 = zero(ẋ2)
xx = rand(n)
uu = rand(m)
ww = rand(r)
dyn_mod(ẋ2,xx,uu,ww)
model.f(_ẋ2,xx,uu,ww)
@test ẋ2 == _ẋ2

_prob = TrajectoryOptimization.Problem(model_d, TrajectoryOptimization.Objective(costfun,N), x0=x0, N=N, dt=dt)
TrajectoryOptimization.initial_controls!(_prob, U0)
rollout!(_prob)
_prob.X[1] = [_prob.X[1]; zeros(_n)]
_prob.U[1] = [_prob.U[1]; _y0]
@test length(_prob.U[1]) == m1

ilqr_solver = AbstractSolver(_prob,opts_ilqr)

@test length(ilqr_solver.Ū[1]) == m1
@test length(ilqr_solver.X̄[1]) == n1
@test size(ilqr_solver.K[1]) == (m1,n1)
@test size(ilqr_solver.d[1]) == (m1,)

rollout!(_prob,ilqr_solver,1.0)
@test ilqr_solver.X̄[1][n .+ (1:_n)] == _y0
@test ilqr_solver.Ū[1][m .+ (1:_n)] == _y0
