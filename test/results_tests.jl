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
pIs = rand(1:10)
pIsN = rand(1:10)
pIc = rand(1:10)
pEs = rand(1:10)
pEsN = pEs + rand(1:10)
pEc = rand(1:10)

n = 10
m = 5
N = 50

r = ConstrainedVectorResults(n,m,N,pIs,pIsN,pIc,pEs,pEsN,pEc)
@test (length(r.gs[1]),length(r.gs)) == (pIs,N)
@test length(r.gs[end]) == pIsN
@test (length(r.gc[1]),length(r.gc)) == (pIc,N)
@test (length(r.hs[1]),length(r.hs)) == (pEs,N)
@test (length(r.hs[end])) == pEsN
@test (length(r.hc[1]),length(r.hc)) == (pEc,N)
@test (size(r.Iμs[1])...,length(r.Iμs)) == (pIs,pIs,N)
@test (size(r.Iμs[end]),length(r.Iμs)) == ((pIsN,pIsN),N)
@test (size(r.Iμc[1])...,length(r.Iμc)) == (pIc,pIc,N)
@test (size(r.Iνs[1]),length(r.Iνs)) == ((pEs,pEs),N)
@test (size(r.Iνs[end]),length(r.Iνs)) == ((pEsN,pEsN),N)
@test (size(r.Iνc[1])...,length(r.Iνc)) == (pEc,pEc,N)

# Test init_results
N = 10
model, obj = Dynamics.dubinscar
obj_con = ConstrainedObjective(obj, u_min=-10, u_max=10)
solver = Solver(model, obj_con, N=N)
solver.opts.infeasible = true
n,m = get_sizes(solver)
pIs, pIsN, pIc, pEs, pEsN, pEc = TrajectoryOptimization.get_num_constraints(solver)
X = rand(n,N)
U = rand(m,N)

results = TrajectoryOptimization.init_results(solver, X, U)
@test TrajectoryOptimization.to_array(results.X) == X
@test TrajectoryOptimization.to_array(results.U)[1:m,:] == U
@test results.λs[1] == zeros(pIs)
@test results.λs[end] == zeros(pIsN)
@test results.λc[1] == zeros(pIc)
@test results.κs[1] == zeros(pEs)
@test results.κs[end] == zeros(pEsN)
@test results.κc[1] == zeros(pEc)


# Test warm start
λs = [i != N ? rand(pIs) : rand(pIsN) for i = 1:N]
λc = [rand(pIc) for i = 1:N]
κs = [i != N ? rand(pEs) : rand(pEsN) for i = 1:N]
κc = [rand(pEc) for i = 1:N]

solver.opts.infeasible = true
results = TrajectoryOptimization.init_results(solver, X, U, λs=λs, λc=λc, κs=κs, κc=κc)
@test TrajectoryOptimization.to_array(results.X) == X
@test TrajectoryOptimization.to_array(results.U)[1:m,:] == U
@test results.λs == λs
@test results.λc == λc
@test results.κs == κs
@test results.κc == κc

# remove infeasible controls
solver.obj.pEc
solver.opts.infeasible = false
results = TrajectoryOptimization.init_results(solver, Array{Float64}(undef,0,0), U, λs=λs, λc=λc, κs=κs, κc=κc)
@test TrajectoryOptimization.to_array(results.U)[1:m,:] == U
@test results.λs == λs
@test results.λc == λc
@test results.κs == κs
@test results.κc == [[] for i = 1:N]
