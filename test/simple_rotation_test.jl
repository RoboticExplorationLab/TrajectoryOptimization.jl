using Parameters
using StaticArrays
using LinearAlgebra
using Rotations
using BenchmarkTools
using MeshCat
using CoordinateTransformations
using GeometryTypes
using TrajOptPlots
const TO = TrajectoryOptimization
import TrajectoryOptimization: Rotation

# Set up visualizer
vis = Visualizer(); open(vis);
dims = [1,1,3]*0.5
robot = HyperRectangle(Vec((-1 .* dims ./2)...), Vec(dims...));
setobject!(vis["robot"], robot);

# Define Spinner Model
@with_kw struct Spinner{R<:Rotation} <: AbstractModel
    mass::Float64 = 1.0
    J::Diagonal{Float64, SVector{3, Float64}} = Diagonal(@SVector ones(3))
end
Base.size(::Spinner{<:UnitQuaternion}) = (7,3)
Base.size(::Spinner) = (7,3)
Dynamics.orientation(::Spinner{UnitQuaternion{T,D}}, x::SVector{N,T2}) where {T,T2,N,D} =
    UnitQuaternion{T2,D}(x[4], x[5], x[6], x[7])
Dynamics.orientation(::Spinner, x::SVector) = R(x[4], x[5], x[6])

function TrajectoryOptimization.dynamics(model::Spinner, x::SVector, u::SVector)
    J = model.J
    ω = @SVector [x[1], x[2], x[3]]
    q = Dynamics.orientation(model, x)
    ωdot = J\(u - ω × (J*ω))
    qdot = kinematics(q,ω)
    return [ωdot; qdot]
end

function TrajectoryOptimization.state_diff_jacobian!(G, model::Spinner, Z::Traj)
    for k in eachindex(Z)
        G[k] = TrajectoryOptimization.state_diff_jacobian(model, state(Z[k]))
    end
end

function TrajectoryOptimization.state_diff_jacobian(::Spinner{<:UnitQuaternion}, x::SVector)
    q = UnitQuaternion(x[4], x[5], x[6], x[7])
    G = TrajectoryOptimization.∇differential(q)
    @SMatrix [1 0 0 0 0 0;
    0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 G[1] G[5] G[ 9];
    0 0 0 G[2] G[6] G[10];
    0 0 0 G[3] G[7] G[11];
    0 0 0 G[4] G[8] G[12]]
end

function TrajectoryOptimization.state_diff_jacobian(
        ::Spinner{UnitQuaternion{T,IdentityMap}}, x::SVector) where T
    return I
end

function TrajectoryOptimization.state_diff(model::Spinner, x, x0)
    δω = @SVector [x[1]-x0[1], x[2]-x0[2], x[3]-x0[3]]
    q  = Dynamics.orientation(model, x)
    q0 = Dynamics.orientation(model, x0)
    inv(q0)*q
    δq = VectorPart(inv(q0)*q)
    return @SVector [δω[1], δω[2], δω[3], δq[1], δq[2], δq[3]]
end

TrajectoryOptimization.state_diff(::Spinner{UnitQuaternion{T,IdentityMap}}, x, x0) where T = x - x0

TrajectoryOptimization.state_diff_size(model::Spinner) = 6
TrajectoryOptimization.state_diff_size(model::Spinner{UnitQuaternion{T,IdentityMap}}) where T = 7

function TrajOptPlots.visualize!(vis, model::Spinner, Z::Traj)
    X = states(Z)
    fps = Int(round(length(Z)/Z[end].t))
    anim = MeshCat.Animation(fps)
    for k in eachindex(Z)
        atframe(anim, k) do
            x = @SVector zeros(3)
            q = UnitQuaternion(Dynamics.orientation(model, X[k]))
            settransform!(vis["robot"], compose(Translation(x), LinearMap(Quat(q))))
        end
    end
    setanimation!(vis, anim)
    return anim
end

model = Spinner{UnitQuaternion{Float64,MRPMap}}()
Dynamics.rotation_type(::Spinner{R}) where R = R

# Params
N = 101
tf = 5.

# Initial and Final conditions
x0 = @SVector [0,0,0, 1,0,0,0.]
xf = @SVector [0,0,0, 1,0,0,0.]


# Intermediate frames
qs = [UnitQuaternion(Quat(RotZ(ang))) for ang in range(0,2pi,length=N)]
X_guess = map(1:N) do k
    om = @SVector [x0[1], x0[2], x0[3]]
    if k > 76
        q = -qs[k]
    else
        q = qs[k]
    end
    [om; SVector(q)]
end

# Build Objective
Q_diag = [(@SVector fill(1e-3,3)); (@SVector fill(1e-1,4))]
R_diag = @SVector fill(1e-4, 3)
no_quat = SVector{6}(deleteat!(collect(1:7), 4))

costs = map(1:N) do k
    x = X_guess[k]
    if Dynamics.rotation_type(model) <: UnitQuaternion{T,IdentityMap} where T
        Q = Diagonal(Q_diag)
    else
        Q = Diagonal(Q_diag[no_quat])
        Gf = TO.state_diff_jacobian(model, x)
        Q = Gf*Q*Gf'
        # Q = Diagonal(Q_diag)
    end
    R = Diagonal(R_diag)

    if k == N
        Q *= 100
    end

    LQRCost(Q,R,x, checks=false)
    # QuadraticCost(Q,R, checks=false)
    # Q = Diagonal(Q_diag[no_quat])
    # RBCost(model, Q, R, x)
    # QuatLQRCost(Q,R,x)
end
obj = Objective(costs)
obj[51].Q

# Problem
model = Spinner{UnitQuaternion{Float64,MRPMap}}()
model = Spinner{UnitQuaternion{Float64,IdentityMap}}()
prob = Problem(model, obj, xf, tf, x0=x0)

# Solve
ilqr = iLQRSolver(prob)
ilqr.opts.verbose = true
U0 = [@SVector zeros(3) for k = 1:N-1]
initial_controls!(ilqr, U0)
solve!(ilqr)
cost(ilqr)
ilqr.K[1]

q = states(ilqr)[N][4:7]
q'xf[4:7]
visualize!(vis, model, ilqr.Z)

# With Quaternion
model_quat = Spinner{UnitQuaternion{VectorPart}}()
Gf = TO.state_diff_jacobian(model_quat, xf)
Q_quat = Gf*Diagonal(diag(Q)[no_quat])*Gf'
no_quat = @SVector [1,2,3,5,6,7]
Qf_quat = Gf*Diagonal(diag(Qf)[no_quat])*Gf'
costfun = QuadraticCost(Q_quat,R, checks=false)
costfun_term = QuadraticCost(Qf_quat, R*0, checks=false)

obj_quat = Objective(costfun, costfun_term, N)

prob_quat = Problem(model_quat, obj_quat, xf, tf, x0=x0)
# Solve
ilqr_quat = iLQRSolver(prob_quat)
ilqr_quat.opts.verbose = true
initial_controls!(ilqr_quat, U0)
solve!(ilqr_quat)
cost(ilqr_quat)
xf[4:7]'states(ilqr_quat)[end][4:7]
visualize!(vis, model, ilqr.Z)


ilqr_quat.opts.verbose = false
@btime begin
    initial_controls!($ilqr_quat, $U0)
    solve!($ilqr_quat)
end

ilqr.opts.verbose = false
@btime begin
    initial_controls!($ilqr, $U0)
    solve!($ilqr)
end


# Step through the solve
solver = iLQRSolver(prob_quat)
initial_controls!(solver, U0)
rollout!(solver)
states(solver)[end]
cost(solver)

Z = get_trajectory(solver)
TO.state_diff_jacobian!(solver.G, solver.model, Z)
TO.discrete_jacobian!(solver.∇F, solver.model, Z)
TO.cost_expansion!(solver.Q, solver.obj, solver.Z)
solver.∇F[50]
solver.Q.x[50]
solver.Q.u[50]
solver.G[50]
ΔV = backwardpass!(solver)
forwardpass!(solver, ΔV, cost(solver))

for k = 1:N
    solver.Z[k].z = solver.Z̄[k].z
end
states(solver)[end]
