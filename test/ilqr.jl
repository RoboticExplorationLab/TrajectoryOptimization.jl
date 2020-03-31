using TrajectoryOptimization
using LinearAlgebra
using TrajOptCore
using RobotDynamics
using BenchmarkTools
using StaticArrays
using Test

solver = iLQRSolver(Problems.Cartpole()...)
n,m,N = size(solver)
dt = solver.Z[1].dt
initialize!(solver)

# Dynamics expansion
RobotDynamics.dynamics_expansion!(solver.D, solver.model, solver.Z)
∇f = zeros(n,n+m)
discrete_jacobian!(RK3, ∇f, solver.model, solver.Z[1])
@test solver.D[1].∇f ≈ ∇f
RobotDynamics.state_diff_jacobian!(solver.G, solver.model, solver.Z)
error_expansion!(solver.D, solver.model, solver.G)
TrajectoryOptimization.static_backwardpass!(solver)
@test error_expansion(solver.D[1], solver.model)[1] ≈ ∇f[1:n,1:n]
@test error_expansion(solver.D[1], solver.model)[2] ≈ ∇f[:,n .+ (1:m)]

# Cost expansion
cost_expansion!(solver.quad_obj, solver.obj, solver.Z)
@test solver.quad_obj[1].Q ≈ solver.obj[1].Q*dt
@test solver.quad_obj[1].q ≈ (solver.obj[1].Q*solver.x0 + solver.obj[1].q)*dt
error_expansion!(solver.Q, solver.quad_obj, solver.model, solver.Z, solver.G)
@test solver.Q[1].Q ≈ solver.obj[1].Q*dt
@test solver.Q[1].q ≈ (solver.obj[1].Q*solver.x0 + solver.obj[1].q)*dt

# Backwardpass
Q = solver.Q[N]
Sxx = SMatrix(Q.Q)
Sx = SVector(Q.q)
@test Sxx ≈ solver.obj[N].Q

k = N-1
Q = TrajOptCore.static_expansion(solver.Q[k])
fdx,fdu = error_expansion(solver.D[k], solver.model)
fdx,fdu = SMatrix(fdx), SMatrix(fdu)
Q = TrajectoryOptimization._calc_Q!(Q, Sxx, Sx, fdx, fdu)
@test Q.xx ≈ solver.obj[1].Q*dt + fdx'Sxx*fdx
Quu_reg = Q.uu
Qux_reg = Q.ux
K_, d_ = TrajectoryOptimization._calc_gains!(solver.K[k], solver.d[k], Quu_reg, Qux_reg, Q.u)
Sxx,Sx,ΔV_ = TrajectoryOptimization._calc_ctg!(Q, K_, d_)

# Whole bp
norm(TrajectoryOptimization.static_backwardpass!(solver) - [-984.2, 492.1], Inf) < 0.01
