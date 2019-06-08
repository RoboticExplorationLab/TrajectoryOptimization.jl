import TrajectoryOptimization: DirectSolver

model = Dynamics.car_model
costfun = Dynamics.car_costfun
N = 51
prob = Problem(rk4(model), Objective(costfun, N), tf=3)
n,m = size(prob)
initial_controls!(prob, ones(m,N-1))
ilqr = iLQRSolverOptions()
solve!(prob, ilqr)

solver = ProjectedNewtonSolver(prob)
NN = length(solver.V.Z)
P = N*n

V = copy(solver.V)
Z = primals(V)
@test V.X[1] == prob.X[1]
V.X[1][1] = 100
@test V[1] == 100
Z[2] = 150
@test V.X[1][2] == 150
@test V[2] == 150
Z .+= 1
@test V[2] == 151
Y = duals(V)
@test length(Y) == P

# Reset
V = solver.V
@test cost(prob, V) == cost(prob)

dynamics_constraints!(prob, solver)
@test maximum(norm.(solver.fVal, Inf)) ≈ 0
dynamics_jacobian!(prob, solver)

cost_expansion!(prob, solver)
H, g = cost_expansion(prob, solver)
D, d = dynamics_expansion(prob, solver)
projection!(prob, solver)

# Gen Newton Functions
mycost, grad_cost, hess_cost, dyn, jacob_dynamics = gen_usrfun_newton(prob)
cost(prob,V) == mycost(Vector(V.Z))
ForwardDiff.gradient(mycost, Vector(V.Z)) ≈ grad_cost(V)
ForwardDiff.hessian(mycost, Vector(V.Z)) ≈ hess_cost(V)
jacob_dynamics(V) ≈ ForwardDiff.jacobian(dyn, Vector(V.Z))
hess_cost(V) ≈ H
grad_cost(V) ≈ g
dyn(V) ≈ d
jacob_dynamics(V) ≈ D
H = Diagonal(H)
Hinv = inv(H)
-D'*((D*Hinv*D')\d)

mycost(V)
norm(grad_cost(V))
norm(dyn(V),Inf)
norm([grad_cost(V); dyn(V)])
V1 = newton_step0(prob, V)
norm(dyn(V1),Inf)
res = grad_cost(V1) + jacob_dynamics(V1)'duals(V1)
norm(res)
mycost(V1)
V1 = newton_step0(prob, V1)
V1.X


# Compare
cost(prob,V)
residual(prob, solver)
δV = newton_step!(prob, solver)
V1 = copy(V)
cost(prob, V1)
V1.V .+= 0.5δV
cost(prob, V1)
dynamics_constraints!(prob, solver, V1)
dynamics_jacobian!(prob, solver, V1)
cost_expansion!(prob, solver, V1)
projection!(prob, solver, V1)
cost(prob, V1)
residual(prob, solver)

g = vcat([[q.x; q.u] for q in solver.Q]...)
ForwardDiff.gradient(v->cost(prob, PrimalDual(v, n,m,N,P)), V.V)

PrimalDual(V.V, n,m,N,P)


V = solver.V
V1 = copy(solver.V)
V1.V .+= δV
cost(prob, V1)


A = [H D'; D zeros(P,P)]
b = Vector([g; d])
δV = -A\b

V + δV
V.Z.indices
@btime copy($V)

cost(prob)
@btime PrimalDual(prob)

struct ConstraintVals{T}
    d::Vector{T}
    D::SparseMatrixCSC{T, Int}
    c::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    ∇c::Vector
end
