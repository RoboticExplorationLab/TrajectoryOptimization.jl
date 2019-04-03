model, obj = Dynamics.quadrotor
N = 21
solver = Solver(model,obj,N=N)
n,m = get_sizes(solver)
U0 = 6*ones(m,N-1)
X0 = rollout(solver,U0)
cost(solver,X0,U0)
res,stats = solve(solver,U0)
stats["cost"][end]
stats["iterations"]
plot(stats["cost"],yscale=:log10)
plot(stats["c_max"])
plot(res.X)

costfun = obj.cost
model_d = Model{Discrete}(model,rk4)
U = to_dvecs(U0)
X = empty_state(n,N)
x0 = obj.x0
dt = solver.dt
# C = AbstractConstraint[]
prob = Problem(model_d,costfun,x0,U,dt)
u_max = 4.0
u_min = 0.0
bnd = bound_constraint(model.n,model.m,u_min=u_min,u_max=u_max,trim=true)
add_constraints!(prob,bnd)

cs = [bnd,bnd]

cs isa ConstraintSet

al_solver = AugmentedLagrangianSolver(prob)
al_prob = AugmentedLagrangianProblem(prob,al_solver)
J = solve!(al_prob,al_solver)
step!()
max_violation(al_solver)
plot(al_prob.U)
plot(al_solver.stats[:c_max])
