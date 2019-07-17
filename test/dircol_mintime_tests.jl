# opts_d = DIRCOLSolverOptions{Float64}()
opts_d_mt = DIRCOLSolverMTOptions(verbose=true,nlp=:Ipopt,R_min_time=1.0,h_max=Inf,h_min=0.0)

model = TrajectoryOptimization.Dynamics.pendulum_model
n = model.n; m = model.m
xf = Problems.pendulum_problem.xf
N = Problems.pendulum_problem.N
tf0 = 2.0
dt = tf0/(N-1)
u_bound = 3.0
bnd = BoundConstraint(n, m, u_min=-u_bound, u_max=u_bound,trim=true)
goal = goal_constraint(xf)

constraints = Constraints([bnd],N)

prob = copy(Problems.pendulum_problem)
prob = update_problem(prob,model=Dynamics.pendulum_model,constraints=constraints)
prob.constraints[N] += goal
# solve!(prob,opts_d)

obj_mt = LQRObjective(Diagonal(zeros(n)),Diagonal(zeros(m)),Diagonal(zeros(n)),xf,N)
prob_mt = update_problem(prob,obj=obj_mt,dt=dt)
copyto!(prob_mt.X,line_trajectory(prob.x0,prob.xf,prob.N))
initial_controls!(prob_mt,[0.01*rand(m) for k = 1:N-1])

plot(prob_mt.X)
solve!(prob_mt,opts_d_mt)
plot(prob_mt.U)


prob = copy(Problems.box_parallel_park_problem)
prob_mt = update_problem(prob,model=Dynamics.car_model)

solve!(prob_mt,opts_d_mt)

plot(prob_mt.U)
