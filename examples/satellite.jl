using RobotZoo
using RobotDynamics
using TrajectoryOptimization
using TrajOptPlots
using TrajOptCore
using StaticArrays
using LinearAlgebra
using Rotations
const TO = TrajectoryOptimization

using MeshCat

if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis)
    set_mesh!(vis, RobotZoo.Satellite())
end

# Discretization
tf = 3.0
N = 101
dt = tf/(N-1)

# Model
model = RobotZoo.Satellite()
n,m = size(model)

# Initial and final states
ω = @SVector zeros(3)
q0 = Rotations.params(UnitQuaternion(I))
qf = Rotations.params(expm(deg2rad(90) * normalize(@SVector rand(3))))
x0 = [ω; q0]
xf = [ω; qf]

# Objective
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(0.1, m))
Qf = (N-1)*Q
obj = LQRObjective(Q,R,Qf,xf,N)

# Solve
prob = Problem(model, obj, xf, tf, x0=x0)
solver = iLQRSolver(prob)
solver.opts.verbose = true
rollout!(solver)
cost(solver)
solve!(solver)
err = rotation_angle(orientation(model, states(solver)[end]) \ orientation(model, xf))
rad2deg(err)

visualize!(vis, solver)

# Use Quat Objective
Q = Diagonal(SA[1,1,1, 0,0,0,0.])
Qf = 100*Q
cost0 = TrajOptCore.QuatLQRCost(Q,  R, xf)
costN = TrajOptCore.QuatLQRCost(Qf, R, xf)
obj = Objective(cost0, costN, N)

prob = Problem(model, obj, xf, tf, x0=x0)
solver = iLQRSolver(prob)
solver.opts.verbose = true
rollout!(solver)
cost(solver)
solve!(solver)
err = rotation_angle(orientation(model, states(solver)[end]) \ orientation(model, xf))
rad2deg(err)

visualize!(vis, solver)
