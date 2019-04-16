using Test, LinearAlgebra
import TrajectoryOptimization: initial_controls!
using BenchmarkTools

# model
T = Float64
model = Dynamics.pendulum_model
n = model.n; m = model.m

# costs
Q = 1e-3*Matrix(I,n,n)
Qf = 100.0*Matrix(I,n,n)
R = 1e-2*Matrix(I,m,m)
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)

costfun = LQRCost(Q, R, Qf, xf)

opts_ilqr = iLQRSolverOptions{T}(cost_tolerance=1.0e-5)
opts_al = AugmentedLagrangianSolverOptions{T}(opts_uncon=opts_ilqr,constraint_tolerance=1.0e-5)

N = 101
dt = 0.1
U0 = [rand(m) for k = 1:N-1]
int_schemes = [:midpoint, :rk3, :rk4]

## Unconstrained
for is in int_schemes
    prob = Problem(model, costfun, integration=is, N=N, dt=dt)
    initial_controls!(prob, U0)
    solver_ilqr = iLQRSolver(prob, opts_ilqr)
    solve!(prob, solver_ilqr)
    @test norm(prob.X[N] - xf) < 1e-6
end

## Constrained
u_bound = 2.
bnd = bound_constraint(n, m, u_min=-u_bound, u_max=u_bound)
goal = goal_constraint(xf)
con = [bnd, goal]

for is in int_schemes
    prob = Problem(model, costfun, integration=is, N=N, dt=dt)
    add_constraints!(prob,con)
    initial_controls!(prob, U0)
    solver_al = AugmentedLagrangianSolver(prob, opts_al)
    solve!(prob, solver_al)
    @test norm(prob.X[N] - xf) < opts_al.constraint_tolerance
    @test max_violation(prob) < opts_al.constraint_tolerance
end

# Test undefined integration
@test_throws ArgumentError Problem(model, costfun, integration=:bogus, N=N)
