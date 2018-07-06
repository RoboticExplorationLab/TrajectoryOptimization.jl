
"""
Midpoint dynamics function for double pendulum
"""
function f(x::Array, u::Array, dt::Float64)
    return x + fc(x + fc(x,u)*dt/2,u)*dt
end

"""
Augmented dynamics function
"""
function fc2(S::Array)
    state = statecache[eltype(S)]
    # state = MechanismState{eltype(S)}(doublependulum)

    # set the state variables:
    set_configuration!(state, S[1:2])
    set_velocity!(state, S[3:4])
    [S[3];S[4]; Array(mass_matrix(state))\S[5:6] - Array(mass_matrix(state))\Array(dynamics_bias(state)); 0.;0.;0.]
end
# function fc2(S::Array)
#     return [fc(S[1:4],S[5:6]); zeros(3)]
# end

function fc2!(out::AbstractVector, S::Array)
    state = statecache[eltype(S)]
    # state = MechanismState{eltype(S)}(doublependulum)

    # set the state variables:
    set_configuration!(state, S[1:2])
    set_velocity!(state, S[3:4])
    copy!(out, [S[3];S[4]; Array(mass_matrix(state))\S[5:6] - Array(mass_matrix(state))\Array(dynamics_bias(state)); 0.;0.;0.])
end

"""
Augmented midpoint dynamics function
"""
function f2(S::Array)
    return S + fc2(S + fc2(S)*S[end]/2)*S[end]
end
Df = S-> ForwardDiff.jacobian(f2, S)
n, m = 4, 2
function f2!(out, S)
    copy!(out, S + fc2!(out, S + fc2!(out, S)*S[end]/2)*S[end])
end
const out = zeros(7)
const result = DiffResults.JacobianResult(out, S)

function f_jacobian(x::Array,u::Array,dt::Float64)
    Df_aug = Df([x;u;dt])
    A = Df_aug[1:n,1:n]
    B = Df_aug[1:n,n+1:n+m]
    return A,B
end

function f_jacobian!(J::Array, x::Array, u::Array, dt::Float64)
    ForwardDiff.jacobian!(result, f2!, out, [x;u;dt])
    copy!(J, DiffResults.jacobian(result))
    A = J[1:n,1:n]
    B = J[1:n,n+1:n+m]
    return A,B
end


# Run f_jacobian
# x0 = zeros(Float64, 4)
# u = [0.;0.]
# dt = 0.1
# @time A1, B1 = f_jacobian(x0, u, dt)
# J = zeros(7,7)
# @time A2, B2 = f_jacobian!(J, x0, u, dt)
