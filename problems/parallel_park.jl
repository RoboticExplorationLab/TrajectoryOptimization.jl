# Box parallel park
T = Float64

# model
model = TrajectoryOptimization.Dynamics.car_model
n = model.n; m = model.m
model_d = rk3(model)

# cost
x0 = [0.; 0.; 0.]
xf = [0.; 1.; 0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-2)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

# constraints
u_bnd = 2.
x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]
bnd = BoundConstraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd,trim=true)

goal_con = goal_constraint(xf)

# problem
N = 51
U = [ones(m) for k = 1:N-1]
obj = LQRObjective(Q,R,Qf,xf,N)
constraints = ProblemConstraints([bnd],N)
dt = 0.06

box_parallel_park_problem = Problem(model_d,obj,U,constraints=copy(constraints),dt=dt,x0=x0,xf=xf)
box_parallel_park_problem.constraints[N] += goal_con
rollout!(box_parallel_park_problem)

box_parallel_park_min_time_problem = Problem(model_d,obj,U,constraints=copy(constraints),dt=dt,x0=x0,xf=xf,tf=:min)
box_parallel_park_min_time_problem.constraints[N] += goal_con
rollout!(box_parallel_park_min_time_problem)
