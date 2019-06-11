
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1,0]
N = 51
n,m = model.n, model.m
bnd = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf], u_min=[0.1,-2], u_max=2)
bnd1 = BoundConstraint(n,m, u_min=bnd.u_min, u_max=bnd.u_max)
goal = goal_constraint(xf)
obs = (([0.2, 0.6], 0.25),
       ([-0.5, 0.5], 0.4))
obs1 = planar_obstacle_constraint(n,m, obs[1]..., :obstacle1)
obs2 = planar_obstacle_constraint(n,m, obs[2]..., :obstacle2)
con = ProblemConstraints(N)
con[1] += bnd1
for k = 2:N-1
    con[k] += bnd + obs1 + obs2
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

solver = ProjectedNewtonSolver(prob)
NN = length(solver.V.Z)
p = solver.p
P = sum(p) + N*n

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
solver.fVal
dynamics_jacobian!(prob, solver)
solver.∇F[1]

cost_expansion!(prob, solver)
H, g = cost_expansion(prob, solver)
D, d = dynamics_expansion(prob, solver)
projection!(prob, solver)

# Gen Newton Functions
mycost, grad_cost, hess_cost, dyn, jacob_dynamics, constraints, jacob_con, act_set =
    gen_usrfun_newton(prob)
cost(prob,V) == mycost(Vector(V.Z))
ForwardDiff.gradient(mycost, Vector(V.Z)) ≈ grad_cost(V)
ForwardDiff.hessian(mycost, Vector(V.Z)) ≈ hess_cost(V)
jacob_dynamics(V) ≈ ForwardDiff.jacobian(dyn, Vector(V.Z))
hess_cost(V) ≈ H
grad_cost(V) ≈ g
dyn(V) ≈ d
jacob_dynamics(V) ≈ Array(D)
H = Diagonal(H)
Hinv = inv(H)
-D'*((D*Hinv*D')\d)

solver.opts.active_set_tolerance = 0.0
act_set(V)
C = constraints(V)
tmp1 = maximum(C)
tmp2 = norm(d, Inf)
findmin(C)
V.active_set[132]
@test max(tmp1, tmp2) == norm([d;C][V.active_set],Inf)
solver.opts.active_set_tolerance = 1e-3
active_set!(prob, solver)
@test max(tmp1, tmp2) < norm([d;C][V.active_set],Inf)

@test length(C) == sum(p)
update_constraints!(prob, solver)
@test C == vcat(solver.C...)
active_set!(prob, solver)
@test all(V.active_set[1:N*n])

@test length(C) == sum(length.(V.λ))
@test N*n == sum(length.(V.ν))
@test length(C) + N*n == length(V.active_set)

jacobian!(solver.∇C, prob.constraints, V.X, V.U)
∇C = jacob_con(V)
@test size(∇C) == (sum(p), NN)
@test cat(solver.∇C..., dims=(1,2)) == ∇C

solver = ProjectedNewtonSolver(prob)
solver.opts.active_set_tolerance = 1e-6
mycost, grad_cost, hess_cost, dyn, jacob_dynamics, constraints, jacob_con, act_set =
    gen_usrfun_newton(prob)
V = solver.V
V_ = copy(V)
act_set(V_,1e-6)
a = V
y = [dyn(V_); constraints(V_)][V_.active_set]
Y = [jacob_dynamics(V_); jacob_con(V_)][V_.active_set,:]
δZ = -Y'*((Y*Y')\y)
V_.Z .+= δZ

act_set(V_,1e-6)
y2 = [dyn(V_); constraints(V_)][V_.active_set]
norm(y2,Inf)
norm(y,Inf)
findmin(y)
NN
cond(Array(Y*Y'))
V.active_set

mycost(V)
norm(grad_cost(V))
norm(dyn(V),Inf)
norm([grad_cost(V); dyn(V)])
V1 = newton_step0(prob, V, 1e-2)
norm(dyn(V1),Inf)
res = grad_cost(V1) + jacob_dynamics(V1)'duals(V1)
norm(res)
mycost(V1)
V1 = newton_step0(prob, V1)
V1.X

res = copy(prob)
projection!(res)
copyto!(res.X, V1.X)
copyto!(res.U, V1.U)
cost(res) < cost(prob)
max_violation(res)
max_violation(prob)

plot()
plot_circle!(obs[1]...)
plot_circle!(obs[2]...)
plot_trajectory!(res.X,markershape=:circle)
plot_trajectory!(res.X)
plot(res.U)


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
