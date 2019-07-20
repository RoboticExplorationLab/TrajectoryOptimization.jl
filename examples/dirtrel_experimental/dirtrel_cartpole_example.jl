model = Dynamics.cartpole_uncertain
n = model.n; m = model.m; r = model.r

# Problem
x0 = [0.;0.;0.;0.]
xf = [0.;pi;0.;0.]

E0 = Diagonal(1.0e-6*ones(n))
H0 = zeros(n,r)
D = Diagonal([4])

N = 101 # knot points

Q = Diagonal(ones(n))
R = Diagonal(0.1*ones(m))
Rh = 0.
Qf = Diagonal(0.0*ones(n))

Q_lqr = Diagonal([10.;10.;1.;1.])
R_lqr = Diagonal(ones(m))
Qf_lqr = Diagonal([100.;100.;100.;100.])

Q_r = Q_lqr
R_r = R_lqr
Qf_r = Qf_lqr

tf0 = 5.
dt = tf0/(N-1)

h_max = dt
h_min = dt

u_max = 10.
u_min = -10.

eig_thr = 1.0e-3

# problem
obj = LQRObjective(Q,R,Qf,xf,N)

goal_con = goal_constraint(xf)
bnd_con = BoundConstraint(n,m,u_min=u_min,u_max=u_max,trim=true)
con = Constraints([bnd_con],N)

prob = TrajectoryOptimization.Problem(model, obj,constraints=con, N=N, tf=tf0, x0=x0, dt=dt)
prob.constraints[N] += goal_con # add goal constraint
copyto!(prob.X,line_trajectory(x0,xf,N)) # initialize state with linear interpolation
copyto!(prob.U,[0.01*rand(m) for k = 1:N-1]) # initialize control with rand

solver = DIRTRELSolver(E0,H0,D,Q,R,Rh,Qf,xf,Q_lqr,R_lqr,Qf_lqr,Q_r,R_r,Qf_r,eig_thr,h_max,h_min)

solve!(prob,solver)

plot(prob.X)
plot(prob.U)
