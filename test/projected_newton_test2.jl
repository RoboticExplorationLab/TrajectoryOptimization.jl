# Solve with ALTRO
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1,0]
N = 51
n,m = model.n, model.m
bnd = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf], u_min=[0.1,-2], u_max=2)
bnd1 = BoundConstraint(n,m, u_min=bnd.u_min)
bnd_x = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf])
goal = goal_constraint(xf)
obs = (([0.2, 0.6], 0.25),
       ([-0.5, 0.5], 0.4))
obs1 = planar_obstacle_constraint(n,m, obs[1]..., :obstacle1)
obs2 = planar_obstacle_constraint(n,m, obs[2]..., :obstacle2)
con = ProblemConstraints(N)
con[1] += bnd1
for k = 2:N-1
    con[k] += bnd1 # + obs1 + obs2
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
# @test !all(solver.a.λ)
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
viol = calc_violations(solver)
@test maximum(maximum.(viol)) == norm(y,Inf)
@test norm(y,Inf) == max_violation(prob)
@test max_violation(solver) == max_violation(prob)

# Test Projection
solver = ProjectedNewtonSolver(prob)
solver.opts.active_set_tolerance = 1e-3
projection!(prob, solver)
update!(prob, solver, solver.V)
max_violation(solver)
multiplier_projection!(prob, solver)

# Build KKT
V0 = copy(solver.V)
cost_expansion!(prob, solver)
J0 = cost(prob, V)
res0 = norm(residual(prob, solver))
viol0 = max_violation(solver)
δV = solveKKT(prob, solver)
V_ = line_search(prob, solver, δV)

solver = ProjectedNewtonSolver(prob)
solver.opts.feasibility_tolerance = 1e-10
V_ = newton_step!(prob, solver)
copyto!(solver.V.V, V_.V)
V_ = newton_step!(prob, solver)

α = 0.5
V_ = V0 + α*δV
@test cost(prob, V_) < J0
update!(prob, solver, V_)
res = norm(residual(prob, solver, V_))
(1-α*0.1)*res0
viol = max_violation(solver)

dynamics_constraints!(prob, solver, V_)
update_constraints!(prob, solver, V_)
dynamics_jacobian!(prob, solver, V_)
constraint_jacobian!(prob, solver, V_)
cost_expansion!(prob, solver, V_)
active_set!(prob, solver)
res = norm(residual(prob, solver, V_))
max_violation(solver)
J = cost(prob, V_)

projection!(prob, solver, V_, false)
dynamics_constraints!(prob, solver, V_)
update_constraints!(prob, solver, V_)
dynamics_jacobian!(prob, solver, V_)
constraint_jacobian!(prob, solver, V_)
J = cost(prob, V_)
res = norm(residual(prob, solver, V_))
viol = max_violation(solver)

J < J0
res < res0

plot_trajectory!(V_.X)
solver.C
calc_violations(solver)


# Old Method
mycost, grad_cost, hess_cost, dyn, jacob_dynamics, constraints, jacob_con, act_set =
    gen_usrfun_newton(prob)
δV0 = newton_step0(prob, V)

cost(prob, V_)
norm(residual(prob, solver, V_))
line_search(prob, solver, δV)
update_constraints!(prob, solver, V_)
max_violation(solver)
projection!(prob, solver, V_, false)
V.V ≈ V0.V
V ≈ V0

norm(residual(prob, solver, V))
norm(residual(prob, solver, V_))
cost(prob, V)
cost(prob, V_)



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
