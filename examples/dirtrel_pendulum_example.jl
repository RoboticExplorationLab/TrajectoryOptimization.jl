model = Dynamics.pendulum_model_uncertain
n = model.n; m = model.m; r = model.r

eig_thr = 1.0e-3
n = 2; m = 1; r = 1

u_max = 3.
u_min = -3.

h_max = Inf
h_min = 0.0

# Problem
x0 = [0.;0.]
xf = [pi;0.]

E0 = Diagonal(1.0e-6*ones(n))
H0 = zeros(n,r)
D = Diagonal([.2^2])

N = 51 # knot points

Q = Diagonal(zeros(n))
R = Diagonal(zeros(m))
Qf = Diagonal(zeros(n))

Q_lqr = Diagonal([10.;1.])
R_lqr = Diagonal(0.1*ones(m))
Qf_lqr = Diagonal([100.;100.])

Q_r = Q_lqr
R_r = R_lqr
Qf_r = Qf_lqr

tf0 = 2.
dt = tf0/(N-1)

# problem
cost_fun = LQRCost(Q,R,Qf,xf)
obj = Objective(cost_fun,N)

goal_con = goal_constraint(xf)
bnd_con = BoundConstraint(n,m,u_min=u_min,u_max=u_max,trim=true)
con = ProblemConstraints([bnd_con],N)

prob = TrajectoryOptimization.Problem(model, obj,constraints=con, N=N, tf=tf0, x0=x0, dt=dt)
prob.constraints[N] += goal_con # add goal constraint
copyto!(prob.X,line_trajectory(x0,xf,N)) # initialize state with linear interpolation
copyto!(prob.U,[0.01*rand(m) for k = 1:N-1]) # initialize control with rand

solver = DIRTRELSolver(E0,H0,D,Q,R,Qf,Q_lqr,R_lqr,Qf_lqr,Q_r,R_r,Qf_r,eig_thr,h_max,h_min)

solve!(prob,solver)

plot(prob.X)
plot(prob.U)
