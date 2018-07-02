
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
# set_configuration!(state, [0., 0.])
# set_configuration!(vis, configuration(state))

"""
Dynamics function for double pendulum
"""
function fc(x::Array{Float64,1},u::Array{Float64,1})
    state = MechanismState{eltype(x)}(doublependulum)

    # set the state variables:
    set_configuration!(state, x[1:2])
    set_velocity!(state, x[3:4])

    # return momentum converted to an `Array` (as this is the format that ForwardDiff expects)
    [x[3];x[4]; Array(mass_matrix(state))\u - Array(mass_matrix(state))\Array(dynamics_bias(state))]
end

"""
Midpoint dynamics function for double pendulum
"""
function f(x::Array,u::Array,dt::Float64)
    return x + fc(x + fc(x,u)*dt/2,u)*dt
end

"""
Augmented dynamics function
"""
function fc2(S::Array)
    state = MechanismState{eltype(S)}(doublependulum)

    # set the state variables:
    set_configuration!(state, S[1:2])
    set_velocity!(state, S[3:4])
    [S[3];S[4]; Array(mass_matrix(state))\S[5:6] - Array(mass_matrix(state))\Array(dynamics_bias(state)); 0.;0.;0.]
end

"""
Augmented midpoint dynamics function
"""
function f2(S::Array)
    return S + fc2(S + fc2(S)*S[end]/2)*S[end]
end
Df = S-> ForwardDiff.jacobian(f2, S)
n, m = 4, 2

function f_jacobian(x::Array,u::Array,dt::Float64)
    Df_aug = Df([x;u;dt])
    A = Df_aug[1:n,1:n]
    B = Df_aug[1:n,n+1:n+m]
    return A,B
end

# Run f_jacobian
x0 = zeros(Float64, 4)
u = [0.;0.]
dt = 0.1
@time f_jacobian(x0, u, dt)

export
    doublependulum,
    state,
    f_jacobian,
    fc

end
