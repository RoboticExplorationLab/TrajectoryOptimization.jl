T = Float64

# model
model = TrajectoryOptimization.Dynamics.pendulum
n = model.n; m = model.m
model_d = rk3(model)

# cost
Q = Array(1e-3*Diagonal(I,n))
R = Array(1e-3*Diagonal(I,m))
Qf = Q
x0 = zeros(n)
xf = [pi;0.0]

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,iterations=50,penalty_scaling=10.0)
opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=15.0,dt_max=0.15,dt_min=1.0e-3)

# constraints
u_bnd = 5.
bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
goal_con = goal_constraint(xf)

# problem
N = 31
U = [ones(m) for k = 1:N-1]
constraints = Constraints(N)
for k = 1:N-1
    constraints[k] += bnd
end
constraints[N] += goal_con
obj = LQRObjective(Q,R,Qf,xf,N)

dt = 0.15
prob = Problem(model_d,obj,U,constraints=constraints,dt=dt,x0=x0)
solve!(prob,opts_altro)
tt = total_time(prob)

PC_mt = TrajectoryOptimization.mintime_constraints(prob)
@test length(PC_mt[1]) == 1
@test length(PC_mt[2]) == 2
@test length(PC_mt[N]) == 2

C = prob.constraints[1]
C2 = TrajectoryOptimization.update_constraint_set_jacobians(C, n, n+1, m)
@test length(C2) == 1

dt = 0.15/2.0
prob_mt = Problem(model_d,obj,prob.U,constraints=constraints,dt=dt,x0=x0,tf=:min)
solve!(prob_mt,opts_altro)
tt_mt = total_time(prob_mt)

n̄ = n+1
idx = [(1:n)...,((1:m) .+ n̄)...]
[collect(1:n);collect(1:m) .+ n̄]

@test tt_mt < 0.5*tt
@test tt_mt < 1.0

@test norm(prob_mt.X[end] - xf,Inf) < 1e-3
@test max_violation(prob_mt) < opts_al.constraint_tolerance

## Box parallel park
model = TrajectoryOptimization.Dynamics.car
n = model.n; m = model.m
model_d = discretize_model(model,:rk4)


# cost
x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-2)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

# options
verbose=false
opts_ilqr = iLQRSolverOptions{T}(verbose=false,live_plotting=:off)
opts_al = AugmentedLagrangianSolverOptions{T}(verbose=false,opts_uncon=opts_ilqr,
    iterations=30,penalty_scaling=10.0)
opts_altro = ALTROSolverOptions{T}(verbose=false,opts_al=opts_al,R_minimum_time=40.,
    dt_max=0.2,dt_min=1.0e-3)

# constraints
u_bnd = 2.
x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]
bnd1 = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
bnd2 = BoundConstraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd)

goal_con = goal_constraint(xf)

# problem
N = 51
U = [ones(m) for k = 1:N-1]
constraints = Constraints(N)
constraints[1] += bnd1
for k = 2:N-1
    constraints[k] += bnd2
end
constraints[N] += goal_con

obj = LQRObjective(Q,R,Qf,xf,N)

dt = 0.06
prob = Problem(model_d,obj,U,constraints=constraints,dt=dt,x0=x0)
solve!(prob,opts_altro)
tt = total_time(prob)

prob_mt = Problem(model_d,obj,prob.U,constraints=constraints,dt=dt,x0=x0,tf=:min)
solve!(prob_mt,opts_altro)
tt_mt = total_time(prob_mt)

@test tt_mt < 0.75*tt
@test tt_mt < 2.1

@test norm(prob_mt.X[end] - xf,Inf) < 1e-3
@test max_violation(prob_mt) < opts_al.constraint_tolerance


########################################
#            PARALLEL PARK             #
########################################


# options
T = Float64
max_con_viol = 1.0e-8
dt_max = 0.2
dt_min = 1.0e-3
verbose=false

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    iterations=30,penalty_scaling=10.0,constraint_tolerance=max_con_viol)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,opts_pn=opts_pn,R_minimum_time=12.5,
    dt_max=dt_max,dt_min=dt_min,projected_newton=true,projected_newton_tolerance=1.0e-4)

opts_ipopt = DIRCOLSolverOptions{T}(verbose=verbose,nlp=:Ipopt,
    opts=Dict(:print_level=>3,:tol=>max_con_viol,:constr_viol_tol=>max_con_viol))

# ALTRO w/ Newton
prob_altro = copy(Problems.parallel_park)
p1, s1 = solve(prob_altro, opts_altro)
@test max_violation_direct(p1) <= 1e-6

# DIRCOL w/ Ipopt
prob_ipopt = copy(Problems.parallel_park)
rollout!(prob_ipopt)
prob_ipopt = update_problem(prob_ipopt,model=Dynamics.car) # get continuous time model
p2, s2 = solve(prob_ipopt, opts_ipopt)
@test max_violation_direct(p2) <= 1e-6

## Minimum Time
max_con_viol = 1.0e-6
opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,live_plotting=:off)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,opts_uncon=opts_ilqr,
    iterations=30,penalty_scaling=10.0,constraint_tolerance=max_con_viol)

opts_pn = ProjectedNewtonSolverOptions{T}(verbose=verbose,feasibility_tolerance=max_con_viol)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,opts_al=opts_al,opts_pn=opts_pn,R_minimum_time=12.5,
    dt_max=dt_max,dt_min=dt_min,projected_newton=true,projected_newton_tolerance=1.0e-4)

opts_mt_ipopt = TO.DIRCOLSolverMTOptions{T}(verbose=verbose,nlp=:Ipopt,
    opts=Dict(:print_level=>0,:tol=>max_con_viol,:constr_viol_tol=>max_con_viol),
    R_min_time=10.0,h_max=dt_max,h_min=dt_min)

# ALTRO w/ Newton
prob_mt_altro = update_problem(copy(Problems.parallel_park),tf=0.) # make minimum time problem by setting tf = 0
initial_controls!(prob_mt_altro,copy(p1.U))
p4, s4 = solve(prob_mt_altro,opts_altro)
@test max_violation_direct(p4) < 1e-6
@test total_time(p4) < 1.6
total_time(p4)


# DIRCOL w/ Ipopt
prob_mt_ipopt = update_problem(copy(Problems.parallel_park))
initial_controls!(prob_mt_ipopt,copy(p2.U))
rollout!(prob_mt_ipopt)
prob_mt_ipopt = update_problem(prob_mt_ipopt,model=Dynamics.car,tf=0.) # get continuous time model
p5, s5 = solve(prob_mt_ipopt, opts_mt_ipopt)
@test max_violation_direct(p5) <= 1e-6
@test total_time(p5) < 1.6
