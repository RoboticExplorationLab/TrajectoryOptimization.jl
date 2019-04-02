using TrajectoryOptimization
using Plots

# Old Method
model, obj = Dynamics.quadrotor
N = 21
solver = Solver(model,obj,N=N)
n,m = get_sizes(solver)
U0 = 6*ones(m,N-1)
X0 = rollout(solver,U0)
J0 = cost(solver,X0,U0)
res,stats = solve(solver,U0)
plot(res.U)

# add constraints
obj_c = ConstrainedObjective(obj,u_min=0,u_max=4.5)
solver = Solver(model,obj_c,N=N)
res_con,stats_con = solve(solver,U0)
plot(res_con.U)

# New Method
model_d = Model{Discrete}(model,rk4)
U = to_dvecs(U0)
X = empty_state(n,N)
dt = solver.dt
x0 = obj.x0
prob = Problem(model_d,costfun,x0,U,dt)

opts = iLQRSolverOptions(iterations=50, gradient_norm_tolerance=1e-4, verbose=false)
ilqr = iLQRSolver(prob,opts)
res1 = solve(prob,ilqr)
plot(res1.U)
