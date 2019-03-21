model, obj = Dynamics.dubinscar
N = 21
solver = Solver(model,obj,N=N)
n,m = get_sizes(solver)
U0 = ones(m,N-1)
res,stats = solve(solver,U0)

costfun = obj.cost
model_d = Model{Discrete}(model,rk4)
prob = Problem(model_d,costfun)
copyto!(prob.U,U0)
ilqr = iLQRSolver(prob)
step!(prob,ilqr)
prob.U isa VectorTrajectory
ilqr.∇F isa PartedMatTrajectory
jacobian!(ilqr.∇F,prob.model,prob.X,prob.U,prob.dt)
