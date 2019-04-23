import TrajectoryOptimization: Model, LQRCost, Problem, Objective, rollout!, iLQRSolverOptions,
    AbstractSolver, jacobian!, _backwardpass!, _backwardpass_sqrt!, AugmentedLagrangianSolverOptions, ALTROSolverOptions,
    bound_constraint, goal_constraint, update_constraints!, update_active_set!, jacobian!, update_problem,line_trajectory

## Pendulum
T = Float64

# model
model = TrajectoryOptimization.Dynamics.pendulum_model
n = model.n; m = model.m
model_d = Model{Discrete}(model,rk4)

# cost
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)
Q = 1e-3*Matrix(I,n,n)
Qf = 1000.0*Matrix(I,n,n)
R = 1e-2*Matrix(I,m,m)
tf = 5.
lqr_cost = LQRCost(Q,R,Qf,xf)

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,constraint_tolerance=1.0e-5,
    cost_tolerance=1.0e-5,cost_tolerance_intermediate=1.0e-5,opts_uncon=opts_ilqr,iterations=30,
    penalty_scaling=10.0)
opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,R_inf=1.0,resolve_feasible_problem=true)

# constraints
u_bnd = 2.
x_min = [-10.;-10.]
x_max = [10.;10.]
bnd = bound_constraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd,trim=true)
bnd = bound_constraint(n,m,u_min=-u_bnd,u_max=u_bnd,trim=true)

goal_con = goal_constraint(xf)
con = [bnd, goal_con]

# problem
N = 51
U = [ones(m) for k = 1:N-1]
dt = 0.1
X0 = line_trajectory(x0,xf,N)

# unconstrained infeasible solve
prob = Problem(model_d,Objective(lqr_cost,N),U,dt=dt,x0=x0)
prob_inf = TrajectoryOptimization.infeasible_problem(prob,1.0)
copyto!(prob.X,X0)
solve!(prob,opts_altro)
@test norm(prob.X[end] - xf) < 1.0e-3

# constrained infeasible solve
prob = Problem(model_d,Objective(lqr_cost,N),U,constraints=ProblemConstraints(con,N),dt=dt,x0=x0)
copyto!(prob.X,X0)
solve!(prob,opts_altro)

@test norm(prob.X[end] - xf) < opts_al.constraint_tolerance
@test max_violation(prob) < opts_al.constraint_tolerance
