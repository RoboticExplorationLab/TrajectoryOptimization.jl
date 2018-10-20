traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf = joinpath(urdf_folder, "cartpole.urdf")

mech = parse_urdf(Float64,urdf)

n = num_positions(mech) + num_velocities(mech) + num_additional_states(mech)
num_joints = length(joints(mech))-1  # subtract off joint to world
m = num_joints # Default to number of joints

torques = [1.0;0.0]
function fc(xdot,x,u)
    state = MechanismState{eltype(x)}(mech)

    # set the state variables:
    q = x[1:num_joints]
    qd = x[1+num_joints:num_joints+num_joints]
    set_configuration!(state, q)
    set_velocity!(state, qd)
    xdot[1:num_joints] = qd
    xdot[num_joints+1:num_joints+num_joints] = Array(mass_matrix(state))\(torques.*u) - Array(mass_matrix(state))\Array(dynamics_bias(state))
    return nothing
end


function fc1(xdot,x,u)
    state1 = MechanismState{eltype(x)}(mech)
    result1 = DynamicsResult(mech)
    dynamics!(xdot, result1, state1, x, torques.*u)
    return nothing
end

xdot1 = zeros(4)
x1 = ones(4)
u1 = ones(2)

xdot2 = zeros(4)
x2 = ones(4)
u2 = ones(2)
using BenchmarkTools
using ForwardDiff

@time for i = 1:1000
    fc(xdot1,x1,u1)
    x1[:] = xdot1*0.1
end

@time for i = 1:1000
    fc1(xdot2,x2,u2)
    x2[:] = xdot2*0.1
end

xdot1




xdot2

f_aug! = f_augmented!(fc1, 4, 2)

Jd = zeros(6, 6)
Sd = zeros(6)
Sdotd = zero(Sd)
Fd!(Jd,Sdotd,Sd) = ForwardDiff.jacobian!(Jd,f_aug!,Sdotd,Sd)

Fd!(Jd,Sdotd,Sd)

state1 = MechanismState(mech)
result1 = DynamicsResult(mech)
f!(xdot,x,u) = dynamics!(xdot, result1, state1, x, u)

f!(xdot1,x1,u1)

f_ip! = f_augmented!(fc1,4,2)

ForwardDiff.jacobian(f_ip!,ones(6),ones(6))

nq = length(result1.q̇)
nv = length(result1.v̇)
ns = length(result1.ṡ)

result1
