module DoublePendulum

using RigidBodyDynamics
using ForwardDiff
using Plots
using MeshCatMechanisms
using iLQR

# Open URDF file
dir = Pkg.dir("DynamicWalking2018")
urdf = joinpath(dir,"notebooks","data","doublependulum.urdf")
doublependulum = parse_urdf(Float64,urdf)

state = MechanismState(doublependulum)
statecache = StateCache(doublependulum)
# set_configuration!(state, [0., 0.])
# set_configuration!(vis, configuration(state))

"""
Dynamics function for double pendulum
"""
function fc(x::Array,u::Array)
    state = statecache[eltype(x)]
    # state = MechanismState{eltype(x)}(doublependulum)

    # set the state variables:
    set_configuration!(state, x[1:2])
    set_velocity!(state, x[3:4])

    # return momentum converted to an `Array` (as this is the format that ForwardDiff expects)
    [x[3];x[4]; Array(mass_matrix(state))\u - Array(mass_matrix(state))\Array(dynamics_bias(state))]
end

export
    doublependulum,
    state,
    f_jacobian,
    f_jacobian!,
    fc

end
