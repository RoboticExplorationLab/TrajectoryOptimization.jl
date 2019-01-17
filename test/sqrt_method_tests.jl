model,obj = TrajectoryOptimization.Dynamics.dubinscar
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
U0 = ones(solver.model.m,solver.N)
results = init_results(solver,Array{Float64}(undef,0,0),U0)
results_sqrt = init_results(solver,Array{Float64}(undef,0,0),U0)
results.X[1] = solver.obj.x0
copyto!(results.U, ones(model.m,solver.N-1))
results_sqrt.X[1] = solver.obj.x0
copyto!(results_sqrt.U, ones(model.m,solver.N-1))

TrajectoryOptimization.rollout!(results,solver)
TrajectoryOptimization.rollout!(results_sqrt,solver)
TrajectoryOptimization.update_jacobians!(results,solver)
TrajectoryOptimization.update_jacobians!(results_sqrt,solver)

n,m,N = TrajectoryOptimization.get_sizes(solver)
m̄,mm = TrajectoryOptimization.get_num_controls(solver)
n̄,nn = TrajectoryOptimization.get_num_states(solver)
bp = TrajectoryOptimization.BackwardPassZOH(nn,mm,solver.N)
bp_sqrt = TrajectoryOptimization.BackwardPassZOH(nn,mm,solver.N)

ΔV = TrajectoryOptimization.backwardpass!(results,solver,bp)

solver.opts.square_root = true
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_sqrt,solver,bp_sqrt)

# test that results from square root backward pass are the same as backward pass
@test all(isapprox.(results.K,results_sqrt.K))
@test isapprox(to_array(results.d),to_array(results_sqrt.d))

S_sqrt = [results_sqrt.S[k]'*results_sqrt.S[k] for k = 1:N]
@test isapprox(to_array(results.S),to_array(S_sqrt))

# backward pass square root for constrained solve
u_min = -10
u_max = 10
obj_c = TrajectoryOptimization.ConstrainedObjective(obj,u_min=u_min,u_max=u_max) # constrained objective
opts_con = TrajectoryOptimization.SolverOptions()
opts_con.square_root = false
opts_con_sqrt = TrajectoryOptimization.SolverOptions()
opts_con_sqrt.square_root = true
solver_con = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts_con)
solver_con_sqrt = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts_con_sqrt)
U0 = ones(solver_con.model.m,solver_con.N)
results_con = init_results(solver,Array{Float64}(undef,0,0),U0)
results_con_sqrt = init_results(solver,Array{Float64}(undef,0,0),U0)
TrajectoryOptimization.update_jacobians!(results_con, solver_con)
TrajectoryOptimization.update_jacobians!(results_con_sqrt, solver_con_sqrt)

n,m,N = TrajectoryOptimization.get_sizes(solver_con)
m̄,mm = TrajectoryOptimization.get_num_controls(solver_con)
n̄,nn = TrajectoryOptimization.get_num_states(solver_con)
bp = TrajectoryOptimization.BackwardPassZOH(nn,mm,solver_con.N)
bp_sqrt = TrajectoryOptimization.BackwardPassZOH(nn,mm,solver_con.N)
TrajectoryOptimization.backwardpass!(results_con, solver_con,bp)
TrajectoryOptimization.backwardpass!(results_con_sqrt, solver_con_sqrt,bp_sqrt)

@test all(isapprox.(results_con.K,results_con_sqrt.K))
@test all(isapprox.(results_con.d,results_con_sqrt.d))
@test all(isapprox.(results_con.s,results_con_sqrt.s))
S_con_sqrt = [results_con_sqrt.S[k]'*results_con_sqrt.S[k] for k = 1:N]
@test isapprox(to_array(results_con.S),to_array(S_con_sqrt))

## Simple pendulum
u_bound = 2.
model, obj = TrajectoryOptimization.Dynamics.pendulum!
obj_c = Dynamics.pendulum_constrained[2]
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false

# unconstrained
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
U = zeros(solver.model.m, solver.N-1)
results, = TrajectoryOptimization.solve(solver,U)
solver.opts.square_root = false
results_sqrt, = TrajectoryOptimization.solve(solver,U)
@test norm(results.X[end]-results_sqrt.X[end]) < 1e-3

# constrained
solver = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts)
results_c, = TrajectoryOptimization.solve(solver, U)
solver.opts.square_root = true
results_c_sqrt, = TrajectoryOptimization.solve(solver, U)
max_c = TrajectoryOptimization.max_violation(results_c_sqrt)
@test norm(results_c.X[end]-results_c_sqrt.X[end]) < 1e-3
@test max_c < 1e-3
