using Parameters
using StaticArrays
using LinearAlgebra
using Rotations
using BenchmarkTools
const TO = TrajectoryOptimization
import TrajectoryOptimization: Rotation

@with_kw struct Spinner{R<:Rotation} <: AbstractModel
    mass::Float64 = 1.0
    J::Diagonal{Float64, SVector{3, Float64}} = Diagonal(@SVector ones(3))
end
Base.size(::Spinner{<:UnitQuaternion}) = (7,3)
Base.size(::Spinner) = (7,3)
Dynamics.orientation(::Spinner{R}, x) where R <: UnitQuaternion = R(x[4], x[5], x[6], x[7])
Dynamics.orientation(::Spinner, x) = R(x[4], x[5], x[6])

function TrajectoryOptimization.dynamics(model::Spinner, x::SVector, u::SVector)
    J = model.J
    ω = @SVector [x[1], x[2], x[3]]
    q = orientation(model, x)
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

function TrajectoryOptimization.state_diff_jacobian(::Spinner{UnitQuaternion{IdentityMap}}, x::SVector)
    return I
end

function TrajectoryOptimization.state_diff(model::Spinner, x, x0)
    δω = @SVector [x[1]-x0[1], x[2]-x0[2], x[3]-x0[3]]
    q  = orientation(model, x)
    q0 = orientation(model, x0)
    inv(q0)*q
    δq = VectorPart(inv(q0)*q)
    return @SVector [δω[1], δω[2], δω[3], δq[1], δq[2], δq[3]]
end

TrajectoryOptimization.state_diff(::Spinner{false}, x, x0) = x - x0

TrajectoryOptimization.state_diff_size(model::Spinner{true}) = 6
TrajectoryOptimization.state_diff_size(model::Spinner{false}) = 7

model = Spinner{false}()

# Params
N = 101
tf = 5.

# Objective
Q = Diagonal(@SVector fill(1e-2,7) )
Qf = 100*Q
R = Diagonal(@SVector fill(0.1, 3))

x0 = @SVector [0,0,0,1,0,0,0.]
qf = Quat(RotZ(pi/4)*RotX(pi/2))
xf = @SVector [0,0,0, qf.w, qf.x, qf.y, qf.z]

# Q = Diagonal(@SVector [0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
# G = TO.state_diff_jacobian(model, xf)
# Q = G*Q*G'
# Qf = 10*Q

obj = LQRObjective(Q,R,Qf,xf,N)

# Problem
prob = Problem(model, obj, xf, tf, x0=x0)

# Solve
ilqr = iLQRSolver(prob)
ilqr.opts.verbose = true
U0 = [@SVector zeros(3) for k = 1:N-1]
initial_controls!(ilqr, U0)
solve!(ilqr)
cost(ilqr)

q = states(ilqr)[N][4:7]
q'xf[4:7]


# With Quaternion
model_quat = Spinner{true}()
Gf = TO.state_diff_jacobian(model_quat, xf)
no_quat = @SVector [1,2,3,5,6,7]
Q_quat = Gf*Diagonal(diag(Q)[no_quat])*Gf'
Qf_quat = Gf*Diagonal(diag(Qf)[no_quat])*Gf'
costfun = QuadraticCost(Q_quat,R)
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

@btime TO.cost!($ilqr_quat.obj, $ilqr_quat.Z)

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
