model, obj = Dynamics.quadrotor
N = 21
solver = Solver(model,obj,N=N)
n,m = get_sizes(solver)
U0 = ones(m,N-1)
X0 = rollout(solver,U0)
cost(solver,X0,U0)
@btime res,stats = solve($solver,$U0)
stats["cost"][end]
plot(res.X)

costfun = obj.cost
model_d = Model{Discrete}(model,rk4)
U = to_dvecs(U0)
X = empty_state(n,N)
x0 = obj.x0
dt = solver.dt
C = AbstractConstraint[]
prob = Problem(model_d,costfun,x0,U,dt)

ilqr = iLQRSolver(prob)
# rollout!(prob)
# plot(prob.X)
# J_prev = cost(prob)
# jacobian!(prob,ilqr)
# ΔV = backwardpass!(prob,ilqr)
# J = forwardpass!(prob,ilqr,ΔV,J_prev)
# step!(prob,ilqr,J_prev)
@btime solve!($prob,$ilqr)
plot(prob.X)

# prob.U isa VectorTrajectory
# ilqr.∇F isa PartedMatTrajectory
# jacobian!(ilqr.∇F,prob.model,prob.X,prob.U,prob.dt)

# all(isfinite.(prob.X[1]))
