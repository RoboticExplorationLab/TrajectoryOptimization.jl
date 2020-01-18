using Test
using StaticArrays
using LinearAlgebra
const TO = TrajectoryOptimization

model_qv = Dynamics.FreeBody{UnitQuaternion{Float64,VectorPart},Float64}()
model_qe = Dynamics.FreeBody{UnitQuaternion{Float64,ExponentialMap},Float64}()
model_qp = Dynamics.FreeBody{UnitQuaternion{Float64,MRPMap},Float64}()
model_qg = Dynamics.FreeBody{UnitQuaternion{Float64,CayleyMap},Float64}()
model_p = Dynamics.FreeBody{MRP{Float64},Float64}()
model_e = Dynamics.FreeBody{RPY{Float64},Float64}()

x,u = rand(model_p)

model_p isa Dynamics.FreeBody{<:Rotation}
size(model_e) == (12,6)
size(model_qv) == (13,6)


# Set up Trajectory Optimization Problem
model = model_qv

# Params
N = 101
tf = 10.
x0 = zeros(model)[1]

r0 = @SVector zeros(3)
q0 = zero(UnitQuaternion)
v0 = @SVector zeros(3)
ω0 = @SVector zeros(3)
x0 = Dynamics.build_state(model, r0, q0, v0, ω0)
@test x0[4] == 1

# Objective
Q = Diagonal(@SVector fill(0.1, 12))
R = Diagonal(@SVector fill(0.05, 6))
Qf = Diagonal(@SVector fill(10.0, 12))

rf = @SVector [10,10,10]
qf = UnitQuaternion(RPY(pi/4, 0, 0))
xf = Dynamics.build_state(model, rf, qf, v0, ω0)
Gf = Dynamics.state_diff_jacobian(model, xf)

Q = Gf*Q*Gf'
Qf = Gf*Qf*Gf'
obj = LQRObjective(Q, R, Qf, xf, N, checks=false)

# Build Problem
prob = Problem(model, obj, xf, tf, x0=x0)

# Solve the problem
ilqr = iLQRSolver(prob)
ilqr.opts.verbose = true
U0 = [@SVector zeros(6) for k = 1:N-1]
initial_controls!(ilqr, U0)
solve!(ilqr)

visualize!(vis, model, get_trajectory(ilqr))

Z = get_trajectory(ilqr)
states(ilqr)[end]
TO.stage_cost(obj[N], Z[end])
controls(ilqr)
ilqr.∇F
ilqr.K

# Step through the solve
solver = ilqr
initial_controls!(solver, U0)
rollout!(solver)
cost(solver)
Z = solver.Z
TO.state_diff_jacobian!(solver.G, solver.model, Z)

solver.G
controls(ilqr)
states(ilqr)[end]


# Visualizer
using MeshCat, GeometryTypes, CoordinateTransformations
vis = Visualizer()
open(vis)
dims = [1,1,3]
robot = HyperRectangle(Vec((-1 .* dims ./2)...), Vec(dims...))
setobject!(vis["robot"], robot)

import TrajectoryOptimization.Dynamics: orientation

Quat(q::UnitQuaternion) = Quat(q.s, q.x, q.y, q.z)
function visualize!(vis, model::RigidBody, Z::Traj)
    X = states(Z)
    fps = Int(round(length(Z)/Z[end].t))
    anim = MeshCat.Animation(fps)
    for k in eachindex(Z)
        atframe(anim, k) do
            x = position(model, X[k])
            q = UnitQuaternion(orientation(model, X[k]))
            settransform!(vis["robot"], compose(Translation(x), LinearMap(Quat(q))))
        end
    end
    setanimation!(vis, anim)
    return anim
end

states(ilqr)[end]
visualize!(vis, model, get_trajectory(ilqr))
Z = get_trajectory(ilqr)
X = states(Z)
states(ilqr)[2]
q = orientation(model, X[end])
settransform!(vis["robot"], LinearMap(Quat(q)))
