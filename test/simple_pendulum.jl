using Test, LinearAlgebra
import TrajectoryOptimization: final_time, initial_controls!
using BenchmarkTools
using Plots

# Set up models and objective
model = Dynamics.pendulum_model
costfun = Dynamics.pendulum_cost
prob0 = Dynamics.pendulum_new
ilqr = iLQRSolverOptions()
ilqr.cost_tolerance = 1e-5
al = AugmentedLagrangianSolverOptions{Float64}(unconstrained_solver=ilqr)
al.constraint_tolerance = 1e-5

n,m = model.n, model.m
N = 501
U0 = rand(m,N-1)
tf = final_time(prob)
xf = [π, 0]
con = prob0.constraints

### UNCONSTRAINED ###
# rk4
prob = Problem(model, costfun, integration=:rk4, N=N, tf=tf)
initial_controls!(prob, U0)
solver = iLQRSolver(prob, ilqr)
solve!(prob, solver)
@test norm(prob.X[N] - xf) < 1e-3


# midpoint
N = 51
prob = Problem(model, costfun, integration=:midpoint, N=N, tf=tf)
@test prob.model.info[:integration] == :midpoint
initial_controls!(prob, U0)
solver = iLQRSolver(prob, ilqr)
solve!(prob, solver)
@test norm(prob.X[N] - xf) < 1e-3
plot(prob.X)


### CONSTRAINED ###
# rk4
bnd = con[1]
prob = Problem(model, costfun, integration=:rk4, constraints=[bnd], N=N, tf=tf)
add_constraints!(prob, goal_constraint(xf))
initial_controls!(prob, U0)
solver = AugmentedLagrangianSolver(prob, al)
# @btime solve($prob, $solver)
solve!(prob, solver)
@test norm(prob.X[N] - xf) < 1e-3
@test max_violation(solver) < 1e-5

# obj_c = Dynamics.pendulum_constrained[2]
# solver = Solver(model,obj_c,dt=0.1,opts=opts)
# results_c, = solve(solver, U0)
# @btime solve($solver, $U0)
# max_c = max_violation(results_c)
# @test norm(results_c.X[end]-xf) < 1e-5
# @test max_c < 1e-5

# midpoint
bnd = con[1]
prob = Problem(model, costfun, integration=:midpoint, constraints=[bnd], N=N, tf=tf)
add_constraints!(prob, goal_constraint(xf))
initial_controls!(prob, U0)
solver = AugmentedLagrangianSolver(prob, al)
@btime solve($prob, $solver)
solve!(prob, solver)
@test norm(prob.X[N] - xf) < 1e-3
@test max_violation(solver) < 1e-5

# solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts)
# results_c, = TrajectoryOptimization.solve(solver, U0)
# max_c = TrajectoryOptimization.max_violation(results_c)
# @test norm(results_c.X[end]-xf) < 1e-5
# @test max_c < 1e-5
# @btime solve($solver, $U0)


# TODO: Change this to new stuff
# Infeasible Start
solver_inf = TrajectoryOptimization.Solver(model, obj_c, dt=0.1, opts=opts)
X_interp = TrajectoryOptimization.line_trajectory(solver_inf)
results_inf, = TrajectoryOptimization.solve(solver_inf,X_interp,U0)
max_c = TrajectoryOptimization.max_violation(results_inf)
@test norm(results_inf.X[end]-xf) < 1e-5
@test max_c < 1e-5

# test linear interpolation for state trajectory
@test norm(X_interp[:,1] - solver_inf.obj.x0) < 1e-8
@test norm(X_interp[:,end] - solver.obj.xf) < 1e-8
@test all(X_interp[1,2:end-1] .<= max(solver.obj.x0[1],solver.obj.xf[1]))
@test all(X_interp[2,2:end-1] .<= max(solver.obj.x0[2],solver.obj.xf[2]))

# test that additional augmented controls can achieve an infeasible state trajectory
obj_c_2 = copy(obj_c)
obj_c_2.x0[:] = ones(solver.model.n)
solver = TrajectoryOptimization.Solver(model, obj_c_2, dt=0.1, opts=opts)
U_infeasible = ones(solver.model.m,solver.N-1)
X_infeasible = ones(solver.model.n,solver.N)
solver.state.infeasible = true  # solver needs to know to use an infeasible rollout
p, pI, pE = TrajectoryOptimization.get_num_constraints(solver::Solver)
p_N, pI_N, pE_N = TrajectoryOptimization.get_num_constraints(solver::Solver)

ui = TrajectoryOptimization.infeasible_controls(solver,X_infeasible,U_infeasible)
results_infeasible = TrajectoryOptimization.ConstrainedVectorResults(solver.model.n,solver.model.m+solver.model.n,p,solver.N,p_N)
copyto!(results_infeasible.U, [U_infeasible;ui])
TrajectoryOptimization.rollout!(results_infeasible,solver)

@test all(ui[1,1:end-1] .== ui[1,1]) # special case for state trajectory of all ones, control 1 should all be same
@test all(ui[2,1:end-1] .== ui[2,1]) # special case for state trajectory of all ones, control 2 should all be same
@test all(TrajectoryOptimization.to_array(results_infeasible.X) == X_infeasible)
# rolled out trajectory should be equivalent to infeasible trajectory after applying augmented controls

### OTHER TESTS ###
# Test undefined integration
@test_throws ArgumentError Problem(model, costfun, integration=:bogus, N=N)
