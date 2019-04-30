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

prob = TrajectoryOptimization.Problem(model, robust_obj, integration=:rk4, x0=x0, N=N, dt=dt)
TrajectoryOptimization.initial_controls!(prob, U_sol)
rollout!(prob)
cost(prob)
prob.obj.robust_cost.ℓw(prob.X,prob.U)
cost(prob.obj.obj,prob.X,prob.U)

prob.obj.robust_cost

prob.obj.robust_cost.∇²ℓw
Qxx, Qux, Quu = unpack_cost_hessian(prob.obj.robust_cost.∇²ℓw,nx,nu,N)

solver_ilqr = TrajectoryOptimization.iLQRSolver(prob, opts_ilqr)

cost_expansion!(prob,solver_ilqr)
cost_expansion!(solver_ilqr.Q,prob.obj.robust_cost,prob.X,prob.U)
solver_ilqr.Q





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
