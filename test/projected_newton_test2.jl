# Solve with ALTRO
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1,0]
N = 51
n,m = model.n, model.m
bnd = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf], u_min=[0.1,-2], u_max=2)
bnd1 = BoundConstraint(n,m, u_min=bnd.u_min, u_max=bnd.u_max)
bnd_x = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf])
goal = goal_constraint(xf)
obs = (([0.2, 0.6], 0.25),
       ([-0.5, 0.5], 0.4))
obs1 = planar_obstacle_constraint(n,m, obs[1]..., :obstacle1)
obs2 = planar_obstacle_constraint(n,m, obs[2]..., :obstacle2)
con = ProblemConstraints(N)
con[1] += bnd1
for k = 2:N-1
    con[k] += bnd # + obs1 + obs2
end
con[N] += goal
prob = Problem(rk4(model), Objective(costfun, N), constraints=con, tf=3)
initial_controls!(prob, ones(m,N-1))
ilqr = iLQRSolverOptions()
al = AugmentedLagrangianSolverOptions(opts_uncon=ilqr)
solve!(prob, al)
plot()
plot_circle!(obs[1]...)
plot_circle!(obs[2]...)
plot_trajectory!(prob.X,markershape=:circle)
plot(prob.U)
max_violation(prob)

# Create PN Solver
solver = ProjectedNewtonSolver(prob)
NN = N*n + (N-1)*m
p = num_constraints(prob)
P = N*n + sum(p)

# Test functions
dynamics_constraints!(prob, solver)
update_constraints!(prob, solver)
active_set!(prob, solver)
@test all(solver.a.primals)
@test all(solver.a.ν)
@test all(solver.a.λ[end-n+1:end])
@test !all(solver.a.λ)
dynamics_jacobian!(prob, solver)
@test solver.∇F[1].xx == solver.Y[1:n,1:n]
@test solver.∇F[2].xx == solver.Y[n .+ (1:n),1:n]
constraint_jacobian!(prob, solver)
@test solver.∇C[1] == solver.Y[N*n .+ (1:4), 1:n+m]

cost_expansion!(prob, solver)
# Y,y = Array(Y), Array(y)

# Test Constraint Violation
solver = ProjectedNewtonSolver(prob)
solver.opts.active_set_tolerance = 0.0
dynamics_constraints!(prob, solver)
update_constraints!(prob, solver)
active_set!(prob, solver)
Y,y = active_constraints(prob, solver)
@test norm(y,Inf) == max_violation(prob)
@test max_violation(solver) == max_violation(prob)

# Test Projection
solver = ProjectedNewtonSolver(prob)
V = solver.V
projection!(prob, solver)
@test max_violation(solver) < max_violation(prob)

# Build KKT
cost_expansion!(prob, solver)
δV = solveKKT(prob, solver)
V_ = solver.V + δV


solver.fVal
solver.C[1] .= 1
@test solver.y[N*n .+ (1:4)] == ones(4)

Y = solver.Y
∇F =
∇C = []

solver isa Vector{PartedArray{T,2,SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}, P} where P} where {T}

solver isa Array{PartedArrays.PartedArray{Float64,2,SubArray{Float64,2,SparseArrays.SparseMatrixCSC{Float64,Int64},Tuple{UnitRange{Int64},UnitRange{Int64}},false},P},1} where P
solver isa
println(typeof(solver))
