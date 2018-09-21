using Test
# Unconstrained Results Constructors
n = rand(1:10)
m = rand(1:10)
N = rand(10:10:100)
r = TrajectoryOptimization.UnconstrainedVectorResults(n,m,N)
@test (length(r.X[1]),length(r.X)) == (n,N)
@test (length(r.U[1]),length(r.U)) == (m,N)
@test (size(r.K[1])...,length(r.K)) == (m,n,N)

r2 = TrajectoryOptimization.UnconstrainedVectorResults(n,m,N)
r.X[1] .= 1:n
copyto!(r2.X,r.X)
@test r2.X[1] == 1:n  # Make sure the copy worked
r2.X[1][1] = 4
@test r.X[1][1] == 1  # Make sure the copies aren't linked


# Static Results
rs = TrajectoryOptimization.UnconstrainedStaticResults(n,m,N)
@test length(rs.X) == N
@test length(rs.X[1]) == n
@test length(rs.U) == N
@test length(rs.U[1]) == m
@test size(rs.K[1]) == (m,n)

# Constrained Results
p = rand(1:5)
p_N = rand(1:5)
r = ConstrainedVectorResults(n,m,p,N,p_N)
@test (length(r.C[1]),length(r.C)) == (p,N)
@test (size(r.Iμ[1])...,length(r.Iμ)) == (p,p,N)


# Static Constrained
r = TrajectoryOptimization.ConstrainedStaticResults(n,m,p,N,p_N)
ax = axes(r.K[1])
r.K[1][ax...]
# maximum(maximum.((r.Iμ .* map((x)->x.>0, r.Iμ))))

# Results Cache (pendulum)
n,m = 2,1
model,obj = TrajectoryOptimization.Dynamics.pendulum
solver = TrajectoryOptimization.Solver(model,obj,dt=0.1)
N = solver.N
N_iter = rand(10:10:100)
rc = TrajectoryOptimization.ResultsCache(solver,N_iter)
@test size(rc.X) == (N,)
@test size(rc.U) == (N,)
@test size(rc.cost) == (N_iter,)
@test_throws MethodError TrajectoryOptimization.ResultsCache(solver,float(N_iter)) # Error on float size

rc = TrajectoryOptimization.ResultsCache(n,m,N,N_iter)
@test size(rc.X) == (N,)
@test size(rc.U) == (N,)
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
    rc.result[i] = TrajectoryOptimization.ConstrainedVectorResults(1,1,1,1)
end
for i = 1:n2
    rc2.result[i] = TrajectoryOptimization.ConstrainedVectorResults(2,2,2,2)
end
R1 = TrajectoryOptimization.ResultsCache(rc.result[1],n1+n2)
R1.result[n1] = copy(rc.result[n1]) # store all valid results

merged = TrajectoryOptimization.merge_results_cache(rc,rc2)
@test size(merged) == length(merged)
@test size(merged) == n1+n2
@test merged.result[1] !== merged.result[2]
@test merged.result[1].X == merged.result[2].X
