using Test
## Pendulum
T = Float64

# model
dyn_pendulum = TrajectoryOptimization.Dynamics.pendulum_dynamics!
n = 2; m = 1
model = Model(dyn_pendulum,n,m)
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
prob = Problem(model_d,lqr_cost,U,dt=dt,x0=x0)
add_constraints!(prob,con)
solve!(prob,opts_altro)
tt = total_time(prob)

dt = 0.15/2.0
prob_mt = Problem(model_d,lqr_cost,U,dt=dt,x0=x0,tf=:min)
add_constraints!(prob_mt,con)
solve!(prob_mt,opts_altro)
prob_mt.U[end]
tt_mt = total_time(prob_mt)

@test tt_mt < 0.5*tt
@test tt_mt < 1.0

@test norm(prob_mt.X[end] - xf) < 1e-3
@test max_violation(prob_mt) < opts_al.constraint_tolerance

## Box parallel park

dyn_car = TrajectoryOptimization.Dynamics.dubins_dynamics!

n = 3; m = 2
model = Model(dyn_car,n,m)
model_d = Model{Discrete}(model,rk4)

# cost
x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)
lqr_cost = LQRCost(Q,R,Qf,xf)

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,
    iterations=30,penalty_scaling=10.0)
opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=1.0,
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
prob = Problem(model_d,lqr_cost,U,dt=dt,x0=x0)
add_constraints!(prob,con)
solve!(prob,opts_altro)
tt = total_time(prob)
plot(prob.U)

prob_mt = Problem(model_d,lqr_cost,U,dt=dt,x0=x0,tf=:min)
add_constraints!(prob_mt,con)
solve!(prob_mt,opts_altro)
plot(prob_mt.U)
tt_mt = total_time(prob_mt)

@test tt_mt < 0.6*tt
@test tt_mt < 1.6

@test norm(prob_mt.X[end] - xf) < 1e-3
@test max_violation(prob_mt) < opts_al.constraint_tolerance
