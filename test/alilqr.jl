using TrajectoryOptimization
using LinearAlgebra
using TrajOptCore
using RobotDynamics
using BenchmarkTools
using StaticArrays
using Test

solver = AugmentedLagrangianSolver(Problems.Cartpole()...)
solve!(solver)
iterations(solver)
cost(solver)
max_violation(solver)
findmax_violation(solver)

n,m,N = size(solver)
initialize!(solver)
ilqr = solver.solver_uncon
initialize!(ilqr)
cost(ilqr)
max_violation(solver)

solve!(ilqr)
iterations(ilqr)
cost(ilqr)
max_violation(solver)

TrajectoryOptimization.dual_update!(solver)
cost(solver)
TrajectoryOptimization.penalty_update!(solver)
cost(solver)


# iLQR solve
TrajectoryOptimization.step!(ilqr, cost(ilqr))
TrajectoryOptimization.copy_trajectories!(ilqr)

RobotDynamics.dynamics_expansion!(ilqr.D, ilqr.model, ilqr.Z)
cost_expansion!(ilqr.quad_obj, ilqr.obj, ilqr.Z)

ilqr.Q === ilqr.quad_obj
ilqr.Q[N].Q
ilqr.Q[N].q
ilqr.Q[N-1].q
ilqr.obj.constraints.λ[2][1] == zeros(n)
ilqr.obj.constraints.μ[2][1] == ones(n)
ilqr.obj.constraints.convals[2].vals[1]
ilqr.D[N-1].A_

TrajectoryOptimization.static_backwardpass!(ilqr)
ilqr.K[1]

ilqr.Q[1].q
ilqr.Q[N].q
ilqr.Q[N-1].q
ilqr.K[N-2]

ΔV = TrajectoryOptimization.static_backwardpass!(ilqr)
TrajectoryOptimization.forwardpass!(ilqr, ΔV, cost(ilqr))
TrajectoryOptimization.copy_trajectories!(ilqr)
max_violation(solver)

ilqr.opts.verbose = true
solve!(ilqr)

TrajOptCore.rese
