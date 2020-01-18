using Parameters
using StaticArrays
using LinearAlgebra

@with_kw struct Spinner <: AbstractModel
    mass::Float64 = 1.0
    J::Diagonal{Float64, SVector{3, Float64}} = Diagonal(@SVector ones(3))
end
Base.size(::Spinner) = (7,3)

function TrajectoryOptimization.dynamics(model::Spinner, x::SVector, u::SVector)
    J = model.J
    ω = @SVector [x[1], x[2], x[3]]
    q = UnitQuaternion(x[4], x[5], x[6], x[7])
    ωdot = J\(u - ω × (J*ω))
    qdot = kinematics(q,ω)
    return [ωdot; qdot]
end

function TrajectoryOptimization.state_diff_jacobian(model::Spinner, x::SVector)
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

function TrajectoryOptimization.state_diff(::Spinner, x, x0)
    δω = @SVector [x[1]-x0[1], x[2]-x0[2], x[3]-x0[3]]
    q = UnitQuaternion(x[4], x[5], x[6], x[7])
    q0 = UnitQuaternion(x[4], x[5], x[6], x[7])
    δq = inv(q0)*q
    return @SVector [δω[1], δω[2], δω[3], δq.x, δq.y, δq.z]
end

TrajectoryOptimization.state_diff_size(::Spinner) = 6

model = Spinner()

# Params
N = 101
tf = 5

# Objective
Q = Diagonal(@SVector [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
Qf = 10*Q
R = Diagonal(@SVector fill(0.1, 3))

x0 = @SVector [0,0,0,1,0,0,0.]
xf = @SVector [0,0,0,sqrt(2)/2,sqrt(2)/2,0,0.]

Q = Diagonal(@SVector [0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
G = state_diff_jacob(xf)
Q = G*Q*G'
Qf = 10*Q

obj = LQRObjective(Q,R,Qf,xf,N)

# Problem
prob = Problem(model, obj, xf, tf, x0=x0)

# Solve
ilqr = iLQRSolver(prob)
ilqr.opts.verbose = true
U0 = [@SVector zeros(3) for k = 1:N-1]
initial_controls!(ilqr, U0)
solve!(ilqr)

q = states(ilqr)[N][4:7]
q'xf[4:7]
