# opts_d = DIRCOLSolverOptions{Float64}()
opts_d_mt = DIRCOLSolverMTOptions(verbose=true,nlp=:Ipopt,R_min_time=1.0,h_max=Inf,h_min=0.0)

model = TrajectoryOptimization.Dynamics.pendulum
n = model.n; m = model.m
xf = Problems.pendulum.xf
N = Problems.pendulum.N
tf0 = 2.0
dt = tf0/(N-1)
u_bound = 3.0
bnd = BoundConstraint(n, m, u_min=-u_bound, u_max=u_bound,trim=true)
goal = goal_constraint(xf)

constraints = Constraints([bnd],N)

prob = copy(Problems.pendulum)
prob = update_problem(prob,model=Dynamics.pendulum,constraints=constraints)
prob.constraints[N] += goal
# solve!(prob,opts_d)

obj_mt = LQRObjective(Diagonal(zeros(n)),Diagonal(zeros(m)),Diagonal(zeros(n)),xf,N)
prob_mt = update_problem(prob,obj=obj_mt,dt=dt)
copyto!(prob_mt.X,line_trajectory(prob.x0,prob.xf,prob.N))
initial_controls!(prob_mt,[0.01*rand(m) for k = 1:N-1])

plot(prob_mt.X)
solve!(prob_mt,opts_d_mt)
plot(prob_mt.U)


prob = copy(Problems.box_parallel_park)
prob_mt = update_problem(prob,model=Dynamics.car)

solve!(prob_mt,opts_d_mt)

plot(prob_mt.U)
