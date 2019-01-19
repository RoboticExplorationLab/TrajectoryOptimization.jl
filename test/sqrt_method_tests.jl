model,obj = TrajectoryOptimization.Dynamics.dubinscar
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
solver_sqrt = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
solver_sqrt.opts.square_root = true
U0 = ones(solver.model.m,solver.N)
results = init_results(solver,Array{Float64}(undef,0,0),U0)
results_sqrt = init_results(solver_sqrt,Array{Float64}(undef,0,0),U0)
results.X[1] = solver.obj.x0
copyto!(results.U, ones(model.m,solver.N-1))
results_sqrt.X[1] = solver_sqrt.obj.x0
copyto!(results_sqrt.U, ones(model.m,solver_sqrt.N-1))

TrajectoryOptimization.rollout!(results,solver)
TrajectoryOptimization.rollout!(results_sqrt,solver_sqrt)
TrajectoryOptimization.update_jacobians!(results,solver)
TrajectoryOptimization.update_jacobians!(results_sqrt,solver_sqrt)

ΔV = TrajectoryOptimization.backwardpass!(results,solver)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_sqrt,solver_sqrt)

@test isapprox(ΔV,ΔV_sqrt)
@test all(isapprox.(results.K,results_sqrt.K))
@test isapprox(to_array(results.d),to_array(results_sqrt.d))
S_sqrt = [results_sqrt.S[k]'*results_sqrt.S[k] for k = 1:solver.N]
@test isapprox(to_array(results.S),to_array(S_sqrt))
max_cn_Quu = backwardpass_max_condition_number(results.bp)
max_cn_S = backwardpass_max_condition_number(results)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_sqrt.bp)
results_sqrt.bp
max_cn_S_sqrt = backwardpass_max_condition_number(results_sqrt)
@test max_cn_Quu_sqrt < max_cn_Quu
@test max_cn_S_sqrt < max_cn_S

# backward pass square root for constrained solve
u_min = -10
u_max = 10
obj_c = TrajectoryOptimization.ConstrainedObjective(obj,u_min=u_min,u_max=u_max) # constrained objective
opts_con = TrajectoryOptimization.SolverOptions()
opts_con_sqrt = TrajectoryOptimization.SolverOptions()
opts_con_sqrt.square_root = true
solver_con = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts_con)
solver_con_sqrt = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts_con_sqrt)
U0 = ones(solver_con.model.m,solver_con.N)
results_con = init_results(solver_con,Array{Float64}(undef,0,0),U0)
results_con_sqrt = init_results(solver_con_sqrt,Array{Float64}(undef,0,0),U0)
TrajectoryOptimization.update_jacobians!(results_con, solver_con)
TrajectoryOptimization.update_jacobians!(results_con_sqrt, solver_con_sqrt)
ΔV = TrajectoryOptimization.backwardpass!(results_con, solver_con)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_con_sqrt, solver_con_sqrt)

@test isapprox(ΔV,ΔV_sqrt)
@test all(isapprox.(results_con.K,results_con_sqrt.K))
@test all(isapprox.(results_con.d,results_con_sqrt.d))
@test all(isapprox.(results_con.s,results_con_sqrt.s))
S_con_sqrt = [results_con_sqrt.S[k]'*results_con_sqrt.S[k] for k = 1:solver_con.N]
@test isapprox(to_array(results_con.S),to_array(S_con_sqrt))
max_cn_Quu = backwardpass_max_condition_number(results_con.bp)
max_cn_S = backwardpass_max_condition_number(results_con)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_con_sqrt.bp)
max_cn_S_sqrt = backwardpass_max_condition_number(results_con_sqrt)
@test max_cn_Quu_sqrt < max_cn_Quu
@test max_cn_S_sqrt < max_cn_S

# square root + infeasible
solver_con_sqrt.state.infeasible = true
U0 = ones(solver_con.model.m,solver_con.N)
X0 = line_trajectory(solver_con)

results_con = init_results(solver_con,X0,U0)
results_con_sqrt = init_results(solver_con_sqrt,X0,U0)
TrajectoryOptimization.update_jacobians!(results_con, solver_con)
TrajectoryOptimization.update_jacobians!(results_con_sqrt, solver_con_sqrt)
ΔV = TrajectoryOptimization.backwardpass!(results_con, solver_con)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_con_sqrt, solver_con_sqrt)

@test isapprox(ΔV,ΔV_sqrt)
@test all(isapprox.(results_con.K,results_con_sqrt.K))
@test all(isapprox.(results_con.d,results_con_sqrt.d))
@test all(isapprox.(results_con.s,results_con_sqrt.s))
S_con_sqrt = [results_con_sqrt.S[k]'*results_con_sqrt.S[k] for k = 1:solver_con.N]
@test isapprox(to_array(results_con.S),to_array(S_con_sqrt))
max_cn_Quu = backwardpass_max_condition_number(results_con.bp)
max_cn_S = backwardpass_max_condition_number(results_con)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_con_sqrt.bp)
max_cn_S_sqrt = backwardpass_max_condition_number(results_con_sqrt)
@test max_cn_Quu_sqrt < max_cn_Quu
@test max_cn_S_sqrt < max_cn_S

# Minimum Time
N = 51
obj_mintime = update_objective(obj_c,tf=:min)
opts_mintime = TrajectoryOptimization.SolverOptions()
opts_mintime_sqrt = TrajectoryOptimization.SolverOptions()
opts_mintime_sqrt.square_root = true
solver_mintime = TrajectoryOptimization.Solver(model,obj_mintime,N=N,opts=opts_mintime)
solver_mintime_sqrt = TrajectoryOptimization.Solver(model,obj_mintime,N=N,opts=opts_mintime_sqrt)
U0 = ones(solver_mintime.model.m,solver_mintime.N)
results_mintime = init_results(solver_mintime,Array{Float64}(undef,0,0),U0)
results_mintime_sqrt = init_results(solver_mintime_sqrt,Array{Float64}(undef,0,0),U0)
TrajectoryOptimization.update_jacobians!(results_mintime, solver_mintime)
TrajectoryOptimization.update_jacobians!(results_mintime_sqrt, solver_mintime_sqrt)
ΔV = TrajectoryOptimization.backwardpass!(results_mintime, solver_mintime)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_mintime_sqrt, solver_mintime_sqrt)

@test all(isapprox.(results_mintime.K,results_mintime_sqrt.K))
@test all(isapprox.(results_mintime.d,results_mintime_sqrt.d))
@test all(isapprox.(results_mintime.s,results_mintime_sqrt.s))
S_mintime_sqrt = [results_mintime_sqrt.S[k]'*results_mintime_sqrt.S[k] for k = 1:solver_mintime.N]
@test isapprox(to_array(results_mintime.S),to_array(S_mintime_sqrt))
max_cn_Quu = backwardpass_max_condition_number(results_mintime.bp)
max_cn_S = backwardpass_max_condition_number(results_mintime)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_mintime_sqrt.bp)
max_cn_S_sqrt = backwardpass_max_condition_number(results_mintime_sqrt)
@test max_cn_Quu_sqrt < max_cn_Quu
@test max_cn_S_sqrt < max_cn_S

# Minimum Time + Infeasible
N = 51
solver_mintime.state.infeasible = true
solver_mintime_sqrt.state.infeasible = true
results_mintime_inf = init_results(solver_mintime,X0,U0)
results_mintime_inf_sqrt = init_results(solver_mintime_sqrt,X0,U0)
TrajectoryOptimization.update_jacobians!(results_mintime_inf, solver_mintime)
TrajectoryOptimization.update_jacobians!(results_mintime_inf_sqrt, solver_mintime_sqrt)
ΔV = TrajectoryOptimization.backwardpass!(results_mintime_inf, solver_mintime)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_mintime_inf_sqrt, solver_mintime_sqrt)

@test isapprox(ΔV,ΔV_sqrt)
@test all(isapprox.(results_mintime_inf.K,results_mintime_inf_sqrt.K))
@test all(isapprox.(results_mintime_inf.d,results_mintime_inf_sqrt.d))
@test all(isapprox.(results_mintime_inf.s,results_mintime_inf_sqrt.s))
S_mintime_inf_sqrt = [results_mintime_inf_sqrt.S[k]'*results_mintime_inf_sqrt.S[k] for k = 1:N]
@test isapprox(to_array(results_mintime_inf.S),to_array(S_mintime_inf_sqrt))
max_cn_Quu = backwardpass_max_condition_number(results_mintime_inf.bp)
max_cn_S = backwardpass_max_condition_number(results_mintime_inf)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_mintime_inf_sqrt.bp)
max_cn_S_sqrt = backwardpass_max_condition_number(results_mintime_inf_sqrt)

@test max_cn_Quu_sqrt < max_cn_Quu
@test max_cn_S_sqrt < max_cn_S

# Terminal cost Hessian .= 0
obj_ = copy(obj)
obj_.cost.Qf .= 0.
obj_.cost.qf .= 0.

opts = TrajectoryOptimization.SolverOptions()
solver = TrajectoryOptimization.Solver(model,obj_,dt=0.1,opts=opts)
solver_sqrt = TrajectoryOptimization.Solver(model,obj_,dt=0.1,opts=opts)
solver_sqrt.opts.square_root = true
U0 = ones(solver.model.m,solver.N)
results = init_results(solver,Array{Float64}(undef,0,0),U0)
results_sqrt = init_results(solver_sqrt,Array{Float64}(undef,0,0),U0)
results.X[1] = solver.obj.x0
copyto!(results.U, ones(model.m,solver.N-1))
results_sqrt.X[1] = solver.obj.x0
copyto!(results_sqrt.U, ones(model.m,solver.N-1))

TrajectoryOptimization.rollout!(results,solver)
TrajectoryOptimization.rollout!(results_sqrt,solver_sqrt)
TrajectoryOptimization.update_jacobians!(results,solver)
TrajectoryOptimization.update_jacobians!(results_sqrt,solver_sqrt)
ΔV = TrajectoryOptimization.backwardpass!(results,solver)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_sqrt,solver_sqrt)

# test that results from square root backward pass are the same as backward pass
@test isapprox(ΔV,ΔV_sqrt)
@test all(isapprox.(results.K,results_sqrt.K))
@test isapprox(to_array(results.d),to_array(results_sqrt.d))
S_sqrt = [results_sqrt.S[k]'*results_sqrt.S[k] for k = 1:N]
@test isapprox(to_array(results.S),to_array(S_sqrt))

max_cn_Quu = backwardpass_max_condition_number(results.bp)
max_cn_S = backwardpass_max_condition_number(results)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_sqrt.bp)
max_cn_S_sqrt = backwardpass_max_condition_number(results_sqrt)

@test max_cn_Quu_sqrt < max_cn_Quu
@test max_cn_S_sqrt < max_cn_S

# Stage cost Q .= 0
obj_ = copy(obj)
obj_.cost.Q .= 0.
obj_.cost.q .= 0.
opts = TrajectoryOptimization.SolverOptions()
solver = TrajectoryOptimization.Solver(model,obj_,dt=0.1,opts=opts)
solver_sqrt = TrajectoryOptimization.Solver(model,obj_,dt=0.1,opts=opts)
solver_sqrt.opts.square_root = true
U0 = ones(solver.model.m,solver.N)
results = init_results(solver,Array{Float64}(undef,0,0),U0)
results_sqrt = init_results(solver_sqrt,Array{Float64}(undef,0,0),U0)
results.X[1] = solver.obj.x0
copyto!(results.U, ones(model.m,solver.N-1))
results_sqrt.X[1] = solver.obj.x0
copyto!(results_sqrt.U, ones(model.m,solver.N-1))

TrajectoryOptimization.rollout!(results,solver)
TrajectoryOptimization.rollout!(results_sqrt,solver_sqrt)
TrajectoryOptimization.update_jacobians!(results,solver)
TrajectoryOptimization.update_jacobians!(results_sqrt,solver_sqrt)
ΔV = TrajectoryOptimization.backwardpass!(results,solver)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_sqrt,solver_sqrt)

# test that results from square root backward pass are the same as backward pass
@test isapprox(ΔV,ΔV_sqrt)
@test all(isapprox.(results.K,results_sqrt.K))
@test isapprox(to_array(results.d),to_array(results_sqrt.d))
S_sqrt = [results_sqrt.S[k]'*results_sqrt.S[k] for k = 1:N]
@test isapprox(to_array(results.S),to_array(S_sqrt))

max_cn_Quu = backwardpass_max_condition_number(results.bp)
max_cn_S = backwardpass_max_condition_number(results)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_sqrt.bp)
max_cn_S_sqrt = backwardpass_max_condition_number(results_sqrt)

@test max_cn_Quu_sqrt < max_cn_Quu
@test max_cn_S_sqrt < max_cn_S

# Stage cost Q .= 0, Terminal Cost Qf .= 0
obj_ = copy(obj)
obj_.cost.Q .= 0.
obj_.cost.q .= 0.
obj_.cost.Qf .= 0.
obj_.cost.qf .= 0.
opts = TrajectoryOptimization.SolverOptions()
solver = TrajectoryOptimization.Solver(model,obj_,dt=0.1,opts=opts)
solver_sqrt = TrajectoryOptimization.Solver(model,obj_,dt=0.1,opts=opts)
solver_sqrt.opts.square_root = true
U0 = ones(solver.model.m,solver.N)
results = init_results(solver,Array{Float64}(undef,0,0),U0)
results_sqrt = init_results(solver_sqrt,Array{Float64}(undef,0,0),U0)
results.X[1] = solver.obj.x0
copyto!(results.U, ones(model.m,solver.N-1))
results_sqrt.X[1] = solver.obj.x0
copyto!(results_sqrt.U, ones(model.m,solver.N-1))

TrajectoryOptimization.rollout!(results,solver)
TrajectoryOptimization.rollout!(results_sqrt,solver_sqrt)
TrajectoryOptimization.update_jacobians!(results,solver)
TrajectoryOptimization.update_jacobians!(results_sqrt,solver_sqrt)
ΔV = TrajectoryOptimization.backwardpass!(results,solver)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_sqrt,solver_sqrt)

# test that results from square root backward pass are the same as backward pass
@test isapprox(ΔV,ΔV_sqrt)
@test norm(to_array(results.K) .- to_array(results_sqrt.K)) < 1e-12
@test norm(to_array(results.d) .- to_array(results_sqrt.d)) < 1e-12
S_sqrt = [results_sqrt.S[k]'*results_sqrt.S[k] for k = 1:N]
@test norm(to_array(results.S) .- to_array(S_sqrt)) < 1e-12

max_cn_Quu = backwardpass_max_condition_number(results.bp)
max_cn_S = backwardpass_max_condition_number(results)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_sqrt.bp)
max_cn_S_sqrt = backwardpass_max_condition_number(results_sqrt)

# NOTE: If you have not state stage costs or state terminal cost the square root method may not be more numerical well conditioned
# @test max_cn_Quu_sqrt max_cn_Quu
# @test max_cn_S_sqrt < max_cn_S # this test fails but that may be ok

# w/ regularization
opts = TrajectoryOptimization.SolverOptions()
opts.bp_reg_initial = 10.0
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
solver_sqrt = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
solver_sqrt.opts.square_root = true
U0 = ones(solver.model.m,solver.N)
results = init_results(solver,Array{Float64}(undef,0,0),U0)
results_sqrt = init_results(solver_sqrt,Array{Float64}(undef,0,0),U0)
results.X[1] = solver.obj.x0
copyto!(results.U, ones(model.m,solver.N-1))
results_sqrt.X[1] = solver.obj.x0
copyto!(results_sqrt.U, ones(model.m,solver.N-1))

TrajectoryOptimization.rollout!(results,solver)
TrajectoryOptimization.rollout!(results_sqrt,solver_sqrt)
TrajectoryOptimization.update_jacobians!(results,solver)
TrajectoryOptimization.update_jacobians!(results_sqrt,solver_sqrt)
ΔV = TrajectoryOptimization.backwardpass!(results,solver)
ΔV_sqrt = TrajectoryOptimization.backwardpass!(results_sqrt,solver_sqrt)

# test that results from square root backward pass are the same as backward pass
@test isapprox(ΔV,ΔV_sqrt)
@test all(isapprox.(results.K,results_sqrt.K))
@test isapprox(to_array(results.d),to_array(results_sqrt.d))
S_sqrt = [results_sqrt.S[k]'*results_sqrt.S[k] for k = 1:N]
@test isapprox(to_array(results.S),to_array(S_sqrt))

max_cn_Quu = backwardpass_max_condition_number(results.bp)
max_cn_S = backwardpass_max_condition_number(results)
max_cn_Quu_sqrt = backwardpass_max_condition_number(results_sqrt.bp)
max_cn_S_sqrt = backwardpass_max_condition_number(results_sqrt)

@test max_cn_Quu_sqrt < max_cn_Quu
@test max_cn_S_sqrt < max_cn_S

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
