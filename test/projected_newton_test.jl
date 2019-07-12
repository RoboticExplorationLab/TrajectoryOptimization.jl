const TO = TrajectoryOptimization
using Test, LinearAlgebra
using ForwardDiff


# Set up Problem
model = Dynamics.car_model
n,m = model.n, model.m
N = 51
Q = Diagonal(I,n)*0.01
R = Diagonal(I,m)*0.01
Qf = Diagonal(I,n)*0.01
xf = [0,1,0]
obj = LQRObjective(Q,R,Qf,xf,N)
n,m = model.n, model.m
bnd = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf], u_min=[0.1,-2], u_max=1.5)
bnd1 = BoundConstraint(n,m, u_min=bnd.u_min)
bnd_x = BoundConstraint(n,m, x_min=[-0.5, -0.01, -Inf], x_max=[0.5, 1.01, Inf])
goal = goal_constraint(xf)
obs = (([0.2, 0.6], 0.25),
       ([-0.5, 0.5], 0.4))
obs1 = TO.planar_obstacle_constraint(n,m, obs[1]..., :obstacle1)
obs2 = TO.planar_obstacle_constraint(n,m, obs[2]..., :obstacle2)
con = Constraints(N)
con[1] += bnd1
for k = 2:N-1
    con[k] += bnd  + obs1 + obs2
end
con[N] += goal
prob = Problem(rk4(model), obj, constraints=con, tf=3)

# Solve with ALTRO
initial_controls!(prob, ones(m,N-1))
ilqr = iLQRSolverOptions()
al = AugmentedLagrangianSolverOptions(opts_uncon=ilqr)
al.constraint_tolerance = 1e-3
al.constraint_tolerance_intermediate = 1e-1
solve!(prob, al)
max_violation(prob)


# Test Primal Dual variable
opts = ProjectedNewtonSolverOptions{Float64}(verbose=true)
solver = ProjectedNewtonSolver(prob,opts)
NN = length(solver.V.Z)
p = num_constraints(prob)
P = sum(p) + N*n


V = copy(solver.V)
Z = TO.primals(V)
@test V.X[1] == prob.X[1]
V.X[1][1] = 100
@test V[1] == 100
Z[2] = 150
@test V.X[1][2] == 150
@test V[2] == 150
Z .+= 1
@test V[2] == 151
Y = TO.duals(V)
@test length(Y) == P

# Create PN Solver
solver = ProjectedNewtonSolver(prob,opts)
NN = N*n + (N-1)*m
p = num_constraints(prob)
P = N*n + sum(p)

# Test functions
TO.dynamics_constraints!(prob, solver)
TO.update_constraints!(prob, solver)
TO.active_set!(prob, solver)
@test all(solver.a.primals)
TO.dynamics_jacobian!(prob, solver)
@test solver.∇F[1].xx == solver.Y[1:n,1:n]
@test solver.∇F[2].xx == solver.Y[n .+ (1:n),1:n]

TO.constraint_jacobian!(prob, solver)
@test solver.∇C[1] == solver.Y[2n .+ (1:p[1]), 1:n+m]

TO.cost_expansion!(prob, solver)


# Check cost gradient and hessian
solver = ProjectedNewtonSolver(prob,opts)
V0 = copy(solver.V.V)
TO.PrimalDual(V0,prob)
TO.cost_expansion!(prob, solver)

function evalcost(V)
    cost(prob, TO.PrimalDual(V,prob))
end
@test evalcost(V0) == cost(prob, solver.V)
@test ForwardDiff.gradient(evalcost, V0)[1:NN] ≈ solver.g
@test solver.H[1:n,1:n] == Q*prob.dt
@test solver.H[n .+ (1:m), n .+ (1:m)] == R*prob.dt
@test solver.H[end-n+1:end, end-n+1:end] == Qf

# Test Constraint Violation
solver = ProjectedNewtonSolver(prob,opts)
solver.opts.active_set_tolerance = 0.0
TO.dynamics_constraints!(prob, solver)
TO.update_constraints!(prob, solver)
TO.dynamics_jacobian!(prob, solver)
TO.constraint_jacobian!(prob, solver)
TO.active_set!(prob, solver)
Y,y = TO.active_constraints(prob, solver)
viol = TO.calc_violations(solver)
@test maximum(maximum.(viol)) == norm(y,Inf)
@test norm(y,Inf) == max_violation(prob)
@test max_violation(solver) == max_violation(prob)

# Test Projection
solver = ProjectedNewtonSolver(prob,opts)
solver.opts.feasibility_tolerance = 1e-10
solver.opts.active_set_tolerance = 1e-3
TO.update!(prob, solver)
Y,y = TO.active_constraints(prob, solver)
TO.projection!(prob, solver)
TO.update!(prob, solver, solver.V)
@test TO.max_violation(solver) < solver.opts.feasibility_tolerance
res0 = norm(TO.residual(prob, solver))
res,δλ = TO.multiplier_projection!(prob, solver)
@test res < res0

solver = ProjectedNewtonSolver(prob,opts)
λ0 = copy(solver.V.Y)
V0 = copy(solver.V.V)
solver.opts.feasibility_tolerance = 1e-10
solver.opts.active_set_tolerance = 1e-3
TO.update!(prob, solver)
Y,y = TO.active_constraints(prob, solver)
TO.primaldual_projection!(prob, solver)
solver.V.Y

TO.update!(prob, solver, solver.V)
@test TO.max_violation(solver) < solver.opts.feasibility_tolerance
res0 = norm(TO.residual(prob, solver))
res, = TO.multiplier_projection!(prob, solver)
@test res < res0
TO.duals(solver.V)[solver.a.duals]
typeof(cholesky(Symmetric(Y*Hinv*Y')))



# Build KKT
Hinv = inv(Diagonal(Array(solver.H)))
TO.cost_expansion!(prob, solver)
J0 = cost(prob, V)
res0 = norm(TO.residual(prob, solver))
@test res0 == res
viol0 = max_violation(solver)
TO.cost_expansion!(prob, solver)
δV, = TO.solveKKT_Shur(prob, solver, Hinv)
V_ = solver.V + 0.5*δV
TO.projection!(prob, solver, V_)
TO.cost_expansion!(prob, solver, V_)
TO.multiplier_projection!(prob, solver, V_)


V_ = TO.line_search(prob, solver, δV)
res = norm(TO.residual(prob, solver, V_))
@test res < res0

# Test Newton Step
solver = ProjectedNewtonSolver(prob,opts)
solver.opts.feasibility_tolerance = 1e-10
solver.opts.verbose = true
V_ = TO.newton_step!(prob, solver)
TO.update!(prob, solver, V_)
@test max_violation(solver) < 1e-10
@test cost(prob, V_) < cost(prob, solver.V)
cost(prob, V_) - cost(prob, solver.V)
@test norm(TO.residual(prob, solver, V_)) < 1e-4

using Plots

plot()
TO.plot_circle!(obs[1]...)
TO.plot_circle!(obs[2]...)
plot_trajectory!(prob.X)
plot_trajectory!(V_.X)

plot(prob.U)
plot!(V_.U)
