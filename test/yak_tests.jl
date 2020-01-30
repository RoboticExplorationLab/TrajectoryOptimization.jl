using TrajectoryOptimization
const TO = TrajectoryOptimization
using TrajOptPlots
using LinearAlgebra
using StaticArrays
using FileIO
using MeshCat
using GeometryTypes
using CoordinateTransformations

# Start visualizer
model = Dynamics.YakPlane()
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

# Discretization
tf = 1.25
dt = 0.025
N = Int(1.25/dt+1)

# Initial and final condition
x0 = @SVector [-3, 0, 1.5, 0.997156, 0, 0.075366, 5, 0, 0, 0, 0, 0]
utrim  = @SVector  [41.6666, 106, 74.6519, 106]
xf = @SVector [3, 0, 1.5, 0, -0.0366076, 0, 5, 0, 0, 0, 0, 0]

# Objective
Qf = Diagonal([(@SVector fill(100,3)); (@SVector fill(500,3)); (@SVector fill(100,6))])
Q = Diagonal(Dynamics.build_state(model, [0,0,0.1], (@SVector fill(0.5,3)), fill(0.1,3), zeros(3)))
R = Diagonal(@SVector fill(1e-3,4))
costfun = LQRCost(Q, R, xf, utrim)
obj = Objective(costfun, N)

# Initialization
U0 = [copy(utrim) for k = 1:N-1]

# Build problem
prob = Problem(model, obj, xf, tf, x0=x0)
solver = iLQRSolver(prob)

initial_controls!(solver, U0)
rollout!(solver)

solver.opts.verbose = true
cost(solver)
solve!(solver)
iterations(solver)
cost(solver)
