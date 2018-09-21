model,obj = TrajectoryOptimization.Dynamics.dubinscar
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
results = TrajectoryOptimization.UnconstrainedVectorResults(model.n,model.m,solver.N)
results_sqrt = TrajectoryOptimization.UnconstrainedVectorResults(model.n,model.m,solver.N)
results.X[1] = solver.obj.x0
copyto!(results.U, ones(model.m,solver.N))
results_sqrt.X[1] = solver.obj.x0
copyto!(results_sqrt.U, ones(model.m,solver.N))

TrajectoryOptimization.rollout!(results,solver)
TrajectoryOptimization.rollout!(results_sqrt,solver)
TrajectoryOptimization.calculate_jacobians!(results,solver)
TrajectoryOptimization.calculate_jacobians!(results_sqrt,solver)

TrajectoryOptimization.backwardpass!(results,solver)
TrajectoryOptimization.backwardpass_sqrt!(results_sqrt,solver)

# test that results from square root backward pass are the same as backward pass
@test all(isapprox.(results.K,results_sqrt.K))
@test all(isapprox.(results.s,results_sqrt.s))
tmp = zero.(results_sqrt.S)
for i = 1:solver.N
    tmp[i] = results_sqrt.S[i]'*results_sqrt.S[i]
end
@test all(isapprox.(results.S,tmp))

# backward pass square root for constrained solve
u_min = -10
u_max = 10
obj_c = TrajectoryOptimization.ConstrainedObjective(obj,u_min=u_min,u_max=u_max) # constrained objective
c_fun, constraint_jacob = TrajectoryOptimization.generate_constraint_functions(obj_c)
_, p = TrajectoryOptimization.is_inplace_constraints(c_fun,n,m)
opts_con = TrajectoryOptimization.SolverOptions()
opts_con.square_root = false
opts_con_sqrt = TrajectoryOptimization.SolverOptions()
opts_con_sqrt.square_root=true
solver_con = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts_con)
solver_con_sqrt = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts_con_sqrt)
results_con = TrajectoryOptimization.ConstrainedVectorResults(model.n,model.m,p[1],solver.N)
results_con_sqrt = TrajectoryOptimization.ConstrainedVectorResults(model.n,model.m,p[1],solver.N)
TrajectoryOptimization.calculate_jacobians!(results_con, solver_con)
TrajectoryOptimization.calculate_jacobians!(results_con_sqrt, solver_con_sqrt)
TrajectoryOptimization.backwardpass!(results_con, solver_con)
TrajectoryOptimization.backwardpass_sqrt!(results_con_sqrt, solver_con_sqrt)
results_con.S
results_con_sqrt.S

@test all(isapprox.(results_con.K,results_con_sqrt.K))
@test all(isapprox.(results_con.s,results_con_sqrt.s))
tmp = zero.(results_con_sqrt.S)
for i = 1:solver.N
    tmp[i] = results_con_sqrt.S[i]''*results_con_sqrt.S[i]
end
@test all(isapprox.(results_con.S,results_con.S))
