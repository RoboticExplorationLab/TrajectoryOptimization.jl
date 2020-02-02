using TrajectoryOptimization
const TO = TrajectoryOptimization
using BenchmarkTools
using TrajOptPlots
using LinearAlgebra
using StaticArrays
using FileIO
using MeshCat
using GeometryTypes
using CoordinateTransformations

# Start visualizer
Rot = MRP{Float64}
model = Dynamics.YakPlane(Rot)
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

# Generate problem
prob = Problems.YakProblems(UnitQuaternion{Float64,IdentityMap})
U0 = deepcopy(controls(prob))

# Solve
solver = iLQRSolver(prob)

initial_controls!(solver, U0)
# benchmark_solve!(solver)
rollout!(solver)
TO.state_diff_size(solver.model)

solver.opts.verbose = true
cost(solver)
solve!(solver)
iterations(solver)
cost(solver)
visualize!(vis, solver)
X = states(solver)
findfirst(isnan.(X))
X[15]


solver.obj.J
