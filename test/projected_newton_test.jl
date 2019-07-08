const TO = TrajectoryOptimization
using Test, LinearAlgebra

# Set up Problem
model = Dynamics.car_model
costfun = Dynamics.car_costfun
xf = [0,1,0]
N = 51
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
prob = Problem(rk4(model), Objective(costfun, N), constraints=con, tf=3)

# Solve with ALTRO
initial_controls!(prob, ones(m,N-1))
ilqr = iLQRSolverOptions()
al = AugmentedLagrangianSolverOptions(opts_uncon=ilqr)
al.constraint_tolerance = 1e-2
al.constraint_tolerance_intermediate = 1e-1
solve!(prob, al)
max_violation(prob)

solver = SequentialNewtonSolver(prob, opts)
solve(prob, solver)



# Test Primal Dual variable
opts = ProjectedNewtonSolverOptions{Float64}(verbose=false)
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
res, = TO.multiplier_projection!(prob, solver)
@test res < res0

# Build KKT
V = solver.V
V0 = copy(V)
TO.cost_expansion!(prob, solver)
J0 = cost(prob, V)
res0 = norm(TO.residual(prob, solver))
viol0 = max_violation(solver)
δV = TO.solveKKT(prob, solver)
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
@test norm(TO.residual(prob, solver, V_)) < 1e-4
