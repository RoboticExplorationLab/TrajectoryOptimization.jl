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
    state = MechanismState{eltype(x)}(mech)
    result = DynamicsResult{eltype(x)}(mech)

    # # set the state variables:
    # q = x[1:num_joints]
    # qd = x[1+num_joints:num_joints+num_joints]
    # set_configuration!(state, q)
    # set_velocity!(state, qd)
    # xdot[1:num_joints] = qd
    # xdot[num_joints+1:num_joints+num_joints] = Array(mass_matrix(state))\(torques.*u) - Array(mass_matrix(state))\Array(dynamics_bias(state))
    dynamics!(xdot, result, state, x, torques.*u)
    return nothing
end

result = DynamicsResult{Float64}(mech)
state = MechanismState(mech)
dynamics!(result, state)

xdot = ones(n)
typeof(result.v̇)

xdot
dynamics!(xdot,result,state,xdot,[1.0;0.0]*10)

xdot

xdot1 = zeros(4)
x1 = ones(4)
u1 = ones(2)

xdot2 = zeros(4)
x2 = ones(4)
u2 = ones(2)

fc(xdot1,x1,u1)

set_configuration!(state, ones(2))
set_velocity!(state, ones(2))
dynamics!(xdot2,result,state,xdot2,torques.*u2)

xdot1

xdot2

state.q
