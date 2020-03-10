using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using Statistics
using TrajOptPlots
using GeometryTypes
using CoordinateTransformations
using BenchmarkTools
using ForwardDiff
using MeshCat
using Test
using InplaceOps
const TO = TrajectoryOptimization
import TrajectoryOptimization.Controllers: RBState

function gen_multisat_prob(Rot=MRP{Float64})
    # Test with rigid bodies
    model = Dynamics.FreeBody{Rot,Float64}()
    models = Dynamics.CopyModel(model, 2)

    # Test solve
    N = 101
    tf = 5.0

    x01 = Dynamics.build_state(model, [0, 1,0], UnitQuaternion(I), zeros(3), zeros(3))
    x02 = Dynamics.build_state(model, [0,-1,0], UnitQuaternion(I), zeros(3), zeros(3))
    x0 = [x01; x02]

    xf1 = Dynamics.build_state(model, [ 1,0,0], expm(deg2rad( 45)*@SVector [1,0,0.]), zeros(3), zeros(3))
    xf2 = Dynamics.build_state(model, [-1,0,0], expm(deg2rad(-45)*@SVector [1,0,0.]), zeros(3), zeros(3))
    xf = [xf1; xf2]

    Qd = Dynamics.fill_state(model, 1e-1, 1e-1, 1e-2, 1e-2)
    Q = Diagonal([Qd; Qd])
    Rd = @SVector fill(1e-2, 6)
    R = Diagonal([Rd; Rd])
    obj = LQRObjective(Q,R,Q*10,xf,N)

    prob = Problem(models, obj, xf, tf, x0=x0)
end
prob = gen_multisat_prob()
solver = iLQRSolver2(prob)
solver.opts.verbose =  false
model = solver.model
x,u = rand(model)

@time rollout!(solver)
@time cost(solver)
@time TO.state_diff_jacobian!(solver.G, solver.model, solver.Z)
@time TO.dynamics_expansion!(solver.D, solver.model, solver.Z)
@time TO.cost_expansion!(solver.Q, solver.obj, solver.Z)
@time TO.error_expansion!(solver.D, solver.model, solver.G)
@time TO.error_expansion!(solver.Q, solver.model, solver.Z, solver.G)
solver.Q[1].xx
solver.G[1]
@time ΔV = TO.static_backwardpass!(solver)
@time TO.forwardpass!(solver, ΔV, cost(solver))

prob = gen_multisat_prob()
solver = iLQRSolver2(prob)
solver.opts.verbose = true
solve!(solver)
cost(solver)



if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model.model)
end
visualize!(vis, model, get_trajectory(solver))
settransform!(vis["robot"]["copy2"], Translation(-1,1,1))
