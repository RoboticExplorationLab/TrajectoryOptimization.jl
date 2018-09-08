
# Unconstrained Results Constructors
n = rand(1:10)
m = rand(1:10)
N = rand(10:10:100)
r = TrajectoryOptimization.UnconstrainedResults(n,m,N)
@test size(r.X) == (n,N)
@test size(r.U) == (m,N)
@test size(r.K) == (m,n,N)

# Results Cache (pendulum)
n,m = 2,1
model,obj = TrajectoryOptimization.Dynamics.pendulum
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1)
N = solver.N
N_iter = rand(10:10:100)
rc = TrajectoryOptimization.ResultsCache(solver,N_iter)
@test size(rc.X) == (n,N)
@test size(rc.U) == (m,N)
@test size(rc.cost) == (N_iter,)
@test_throws MethodError c = TrajectoryOptimization.ResultsCache(solver,float(N_iter)) # Error on float size

rc = TrajectoryOptimization.ResultsCache(n,m,N,N_iter)
@test size(rc.X) == (n,N)
@test size(rc.U) == (m,N)
@test size(rc.cost) == (N_iter,)

# Merge caches
res, = TrajectoryOptimization.solve(solver)
rc = TrajectoryOptimization.ResultsCache(res,10)
TrajectoryOptimization.add_iter!(rc,res,10.,0.,1) # Add results to cache
@test rc.cost[1] == 10
@test rc.time[1] == 0
@test rc.result[1].X == res.X # Make sure values are the same
@test !(rc.result[1].X === res.X) # Make sure the references are different

# Run constrained problem
obj_c = TrajectoryOptimization.ConstrainedObjective(obj,u_max=2.,u_min=-2.)
solver_c = TrajectoryOptimization.Solver(model,obj_c,dt=0.1)
res, = TrajectoryOptimization.solve(solver_c)

# Cache constrained results
rc.result[1].X
rc.result
TrajectoryOptimization.add_iter!(rc,res,10.,0.,2) #TODO fix the overloaded length method
@test rc.result[1].X !== rc.result[2] # Make sure they're not equal

# Cache in another cache
rc2 = TrajectoryOptimization.ResultsCache(solver_c,5)
TrajectoryOptimization.add_iter!(rc2,res,20.,1.,1)
@test rc.result[2].X == rc2.result[1].X
@test !(rc.result[2].X === rc2.result[1].X)

#Fill and  Merge Caches
n1 = rc.termination_index
n2 = rc2.termination_index
for i = 1:n1
    rc.result[i] = TrajectoryOptimization.ConstrainedResults(1,1,1,1)
end
for i = 1:n2
    rc2.result[i] = TrajectoryOptimization.ConstrainedResults(2,2,2,2)
end
R1 = TrajectoryOptimization.ResultsCache(rc.result[1],n1+n2)
i = n1
R1.result[i] = copy(rc.result[i]) # store all valid results

merged = TrajectoryOptimization.merge_results_cache(rc,rc2)
@test size(merged) == length(merged)
@test size(merged) == n1+n2
@test merged.result[1] !== merged.result[2]
@test merged.result[1].X == merged.result[2].X
