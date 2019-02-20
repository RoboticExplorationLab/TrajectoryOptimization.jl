using Test
using BenchmarkTools
using Plots
using SparseArrays

model,obj = Dynamics.quadrotor
model,obj_con = Dynamics.quadrotor_3obs
##########
N = 101 # 201
integration = :rk4
opts = SolverOptions()
opts.verbose = false
opts.square_root = false
opts.cost_tolerance = 1e-5
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-4
opts.outer_loop_update_type = :feedback


# Obstacle Avoidance
model,obj_uncon = TrajectoryOptimization.Dynamics.quadrotor
r_quad = 3.0
n = model.n
m = model.m
obj_con = TrajectoryOptimization.Dynamics.quadrotor_3obs[2]
spheres = TrajectoryOptimization.Dynamics.quadrotor_3obs[3]
n_spheres = length(spheres)

solver_uncon = Solver(model,obj_uncon,integration=integration,N=N,opts=opts)
solver_con = Solver(model,obj_con,integration=integration,N=N,opts=opts)

U_hover = 0.5*9.81/4.0*ones(solver_uncon.model.m, solver_uncon.N-1)
X_hover = rollout(solver_uncon,U_hover)
# @time results_uncon, stats_uncon = solve(solver_uncon,U_hover)
# @time results_uncon_dircol, stats_uncon_dircol = TrajectoryOptimization.solve_dircol(solver_uncon, X_hover, U_hover, options=dircol_options)

@time res, stats_con = solve(solver_con,U_hover)
solver = solver_con
#########

n = 13
m = 4

# Unconstrained
Q = (1e-3)*Matrix(I,n,n)
R = (1e-2)*Matrix(I,m,m)
# R = (1e-1)*Matrix(I,m,m)
Qf = (1000.0)*Matrix(I,n,n)
tf = 1.0
dt = 0.1

# -initial state
x0 = zeros(n)
x0[1:3] = [0.; 0.; 0.]
q0 = [1.;0.;0.;0.]
x0[4:7] = q0

# -final state
xf = copy(x0)
xf[1:3] = [0.;10.;0.] # xyz position
xf[4:7] = q0
u_max = 10.0
u_min = 0.0

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max)#,u_min=u_min)
solver = Solver(model,obj_con,N=101)

solver.obj
solver.opts.cost_tolerance = 1e-5
solver.opts.constraint_tolerance = 1e-4
solver.opts.penalty_max = 1e8
n,m,N = get_sizes(solver)
Random.seed!(1)
U = 0.5*9.81/4.0*ones(solver.model.m, solver.N-1)
res,stats = solve(solver,U)
plot(res.U)
plot(res.X)


cost(solver,res)
λ_update_default!(res,solver);
update_constraints!(res,solver)
max_violation(res)
@assert max_violation(res) < opts.constraint_tolerance

J_prev = cost(solver,res)

p,pI,pE = get_num_constraints(solver)
p_N,pI_N,pE_N = get_num_terminal_constraints(solver)

## Newton 2 ###############
results_new = copy(res)
newton_results = NewtonResults(solver)
newton_active_set!(newton_results,results_new,solver)
update_newton_results!(newton_results,results_new,solver)
newton_step!(results_new,newton_results,solver,1.0)
max_violation(results_new)
newton_cost(res,newton_results,solver)

# newton_solve!(res,solver)
