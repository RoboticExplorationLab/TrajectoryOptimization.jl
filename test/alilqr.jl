using TrajectoryOptimization
using LinearAlgebra
using TrajOptCore
using RobotDynamics
using BenchmarkTools
using StaticArrays
using Test

# Test whole solve
solver = AugmentedLagrangianSolver(Problems.Cartpole()...)
solve!(solver)
@test iterations(solver) == 39
@test abs(cost(solver) - 1.563) < 0.01
@test max_violation(solver) < 0.001
@test findmax_violation(solver) == "GoalConstraint at time step 101 at index 1"

# Test single iLQR solve
solver = AugmentedLagrangianSolver(Problems.Cartpole()...)
n,m,N = size(solver)
initialize!(solver)
ilqr = solver.solver_uncon
initialize!(ilqr)
@test abs(cost(ilqr) - 500) < 1
@test abs(max_violation(solver) - 3.1419) < 0.001

solve!(ilqr)
@test iterations(ilqr) == 70
@test abs(cost(ilqr) - 1.498) < 0.001
@test max_violation(solver) < 0.1

J = cost(solver)
TrajectoryOptimization.dual_update!(solver)
@test cost(solver) > J
TrajectoryOptimization.penalty_update!(solver)
@test cost(solver) > J


# iLQR solve
solver = AugmentedLagrangianSolver(Problems.Cartpole()...)
ilqr = solver.solver_uncon
initialize!(ilqr)
TrajectoryOptimization.step!(ilqr, cost(ilqr))
TrajectoryOptimization.copy_trajectories!(ilqr)

RobotDynamics.dynamics_expansion!(ilqr.D, ilqr.model, ilqr.Z)
cost_expansion!(ilqr.quad_obj, ilqr.obj, ilqr.Z)

@test ilqr.Q === ilqr.quad_obj
@test ilqr.Q[N].Q ≈ diagm(fill(101,n))
@test norm(ilqr.obj.constraints.λ[2][1]) ≈ 0
@test ilqr.obj.constraints.μ[2][1] == ones(n)

@test norm(TrajectoryOptimization.static_backwardpass!(ilqr) - [-559.8, 279.9]) < 0.1
@test norm(ilqr.d[1] - [1.52]) < 0.001
