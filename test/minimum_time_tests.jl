import TrajectoryOptimization: Model, LQRCost, Problem, Objective, rollout!, iLQRSolverOptions,
    AbstractSolver, jacobian!, _backwardpass!, _backwardpass_sqrt!, AugmentedLagrangianSolverOptions, ALTROSolverOptions,
    bound_constraint, goal_constraint, update_constraints!, update_active_set!, jacobian!, update_problem,
    line_trajectory_new, total_time

T = Float64

# model
model = TrajectoryOptimization.Dynamics.pendulum_model
n = model.n; m = model.m
model_d = Model{Discrete}(model,rk4)

# cost
Q = Array(1e-3*Diagonal(I,n))
R = Array(1e-3*Diagonal(I,m))
Qf = Array(Diagonal(I,n)*0.0)
x0 = zeros(n)
xf = [pi;0.0]
lqr_cost = LQRCost(Q,R,Qf,xf)

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,iterations=50,penalty_scaling=2.0)
opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=15.0,dt_max=0.15,dt_min=1.0e-3)

# constraints
u_bnd = 5.
bnd = bound_constraint(n,m,u_min=-u_bnd,u_max=u_bnd,trim=true)
bnd
goal_con = goal_constraint(xf)
con = [bnd, goal_con]

# problem
N = 31
U = [ones(m) for k = 1:N-1]
dt = 0.15
prob = Problem(model_d,Objective(lqr_cost,N),U,constraints=ProblemConstraints(con,N),dt=dt,x0=x0)
solve!(prob,opts_altro)
tt = total_time(prob)

dt = 0.15/2.0
prob_mt = Problem(model_d,Objective(lqr_cost,N),U,constraints=ProblemConstraints(con,N),dt=dt,x0=x0,tf=:min)
solve!(prob_mt,opts_altro)
prob_mt.U[end][end]
tt_mt = total_time(prob_mt)

@test tt_mt < 0.5*tt
@test tt_mt < 1.0

@test norm(prob_mt.X[end] - xf) < 1e-3
@test max_violation(prob_mt) < opts_al.constraint_tolerance

## Box parallel park
model = TrajectoryOptimization.Dynamics.car_model
n = model.n; m = model.m
model_d = Model{Discrete}(model,rk4)

# cost
x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-2)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)
lqr_cost = LQRCost(Q,R,Qf,xf)

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,
    iterations=30,penalty_scaling=10.0)
opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=15.0,
    dt_max=0.2,dt_min=1.0e-3)

# constraints
u_bnd = 2.
x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]
bnd = bound_constraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd,trim=true)

goal_con = goal_constraint(xf)
con = [bnd, goal_con]

# problem
N = 51
U = [ones(m) for k = 1:N-1]
dt = 0.06
prob = Problem(model_d,Objective(lqr_cost,N),U,constraints=ProblemConstraints(con,N),dt=dt,x0=x0)
solve!(prob,opts_altro)
tt = total_time(prob)

prob_mt = Problem(model_d,Objective(lqr_cost,N),U,constraints=ProblemConstraints(con,N),dt=dt,x0=x0,tf=:min)
solve!(prob_mt,opts_altro)
tt_mt = total_time(prob_mt)

@test tt_mt < 0.75*tt
@test tt_mt < 2.1

@test norm(prob_mt.X[end] - xf) < 1e-3
@test max_violation(prob_mt) < opts_al.constraint_tolerance
