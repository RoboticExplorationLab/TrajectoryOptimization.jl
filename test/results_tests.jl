using Test
# Unconstrained Results Constructors
n = rand(1:10)
m = rand(1:10)
N = rand(10:10:100)
r = TrajectoryOptimization.UnconstrainedVectorResults(n,m,N)
@test (length(r.X[1]),length(r.X)) == (n,N)
@test (length(r.U[1]),length(r.U)) == (m,N-1)
@test (size(r.K[1])...,length(r.K)) == (m,n,N-1)

r2 = TrajectoryOptimization.UnconstrainedVectorResults(n,m,N)
r.X[1] .= 1:n
copyto!(r2.X,r.X)
@test r2.X[1] == 1:n  # Make sure the copy worked
r2.X[1][1] = 4
@test r.X[1][1] == 1  # Make sure the copies aren't linked


# Constrained Results
p = rand(1:5)
p_N = rand(1:5)
r = ConstrainedVectorResults(n,m,p,N,p_N,TrajectoryVariable)
@test (length(r.C[1]),length(r.C)) == (p,N)
@test (size(r.Iμ[1])...,length(r.Iμ)) == (p,p,N)


# Test init_results
N = 10
model, obj = Dynamics.dubinscar
obj_con = ConstrainedObjective(obj, u_min=-10, u_max=10)
solver = Solver(model, obj_con, N=N)
solver.state.infeasible = true
n,m = get_sizes(solver)
p, = get_num_constraints(solver)
p_N, = TrajectoryOptimization.get_num_terminal_constraints(solver)
X = rand(n,N)
U = rand(m,N-1)

results = init_results(solver, X, U)
@test isapprox(to_array(results.X),X)
@test isapprox(to_array(results.U)[1:m,:],U)
@test results.λ[1] == zeros(p)

# Test warm start (partial ⁠λs)
λ = [i != N ? rand(p-n) : rand(p_N) for i = 1:N]
results = init_results(solver, X, U, λ=λ)
@test isapprox(to_array(results.X),X)
@test isapprox(to_array(results.U)[1:m,:],U)
@test results.λ[1] == [λ[1]; zeros(n)]
λ[1][1] = 10
@test results.λ[1][1] != 10

# Test warm start (all ⁠λs)
λ = [rand(p) for i = 1:N]
push!(λ,rand(n))
results = init_results(solver, X, U, λ=λ)
@test isapprox(to_array(results.X),X)
@test isapprox(to_array(results.U)[1:m,:],U)
@test results.λ[1] == λ[1]
λ[1][1] = 10
@test results.λ[1][1] != 10

λ = [rand(p-1) for i = 1:N-1]
push!(λ,rand(n))
@test_throws ArgumentError init_results(solver, X, U, λ=λ)
