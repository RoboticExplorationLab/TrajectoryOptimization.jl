using TrajectoryOptimization
using Test

model,obj = TrajectoryOptimization.Dynamics.dubinscar
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = true
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1,opts=opts)
results = TrajectoryOptimization.UnconstrainedResults(model.n,model.m,solver.N)
results_sqrt = TrajectoryOptimization.UnconstrainedResults(model.n,model.m,solver.N)
results.X[:,1] = solver.obj.x0
results.U[:,:] = ones(model.m,solver.N)
results_sqrt.X[:,1] = solver.obj.x0
results_sqrt.U[:,:] = ones(model.m,solver.N)

TrajectoryOptimization.rollout!(results,solver)
TrajectoryOptimization.rollout!(results_sqrt,solver)
TrajectoryOptimization.calc_jacobians!(results,solver)
TrajectoryOptimization.calc_jacobians!(results_sqrt,solver)

TrajectoryOptimization.backwardpass!(results,solver)
TrajectoryOptimization.backwards_sqrt!(results_sqrt,solver)

# test that results from square root backward pass are the same as backward pass
@test all(isapprox.(results.K,results_sqrt.K))
@test all(isapprox.(results.s,results_sqrt.s))
tmp = zeros(size(results_sqrt.S))
for i = 1:solver.N
 tmp[:,:,i] = results_sqrt.S[:,:,i]'*results_sqrt.S[:,:,i]
end
@test all(isapprox.(results.S,tmp))

# backward pass square root for constrained solve
u_min = -10
u_max = 10
obj_c = TrajectoryOptimization.ConstrainedObjective(obj,u_min=u_min,u_max=u_max) # constrained objective
c_fun, constraint_jacobian = TrajectoryOptimization.generate_constraint_functions(obj_c)
p = size(c_fun(zeros(solver.model.n),zeros(solver.model.m)))
opts_con = TrajectoryOptimization.SolverOptions()
opts_con.square_root = false
opts_con_sqrt = TrajectoryOptimization.SolverOptions()
opts_con_sqrt.square_root=true
solver_con = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts_con)
solver_con_sqrt = TrajectoryOptimization.Solver(model,obj_c,dt=0.1,opts=opts_con_sqrt)
results_con = TrajectoryOptimization.ConstrainedResults(model.n,model.m,p[1],solver.N)
results_con_sqrt = TrajectoryOptimization.ConstrainedResults(model.n,model.m,p[1],solver.N)
TrajectoryOptimization.calc_jacobians!(results_con, solver_con)
TrajectoryOptimization.calc_jacobians!(results_con_sqrt, solver_con_sqrt)
TrajectoryOptimization.backwardpass!(results_con, solver_con)
TrajectoryOptimization.backwards_sqrt!(results_con_sqrt, solver_con_sqrt)
results_con.S
results_con_sqrt.S

@test all(isapprox.(results_con.K,results_con_sqrt.K))
@test all(isapprox.(results_con.s,results_con_sqrt.s))
tmp = zeros(size(results_con_sqrt.S))
for i = 1:solver.N
 tmp[:,:,i] = results_con_sqrt.S[:,:,i]''*results_con_sqrt.S[:,:,i]
end
@test all(isapprox.(results_con.S,results_con.S))
