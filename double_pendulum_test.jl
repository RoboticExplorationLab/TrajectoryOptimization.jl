using iLQR
using RigidBodyDynamics
using DoublePendulum
using ForwardDiff

## Problem Setup
n = 4
m = 2

# initial and goal states
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

set_configuration!(state, x0[1:2])
set_velocity!(state, x0[3:4])

# costs
Q = 1e-5*eye(n)
Qf = 25.*eye(n)
R = 1e-5*eye(m)

# simulation
tf = 1.
dt = 0.1

# objects
model = iLQR.Model(doublependulum)
obj = iLQR.Objective(Q, R, Qf, tf, x0, xf)

q = [1;2]
qd = [3;4]
u = [5.;2.]
dt = 0.1
S = [q;qd;u;dt]
x = [q;qd]


fd = iLQR.f_midpoint(model.f, dt)     # Discrete dynamics
f_aug = iLQR.f_augmented(model)  # Augmented continuous dynamics
fd_aug = iLQR.f_midpoint(f_aug)  # Augmented discrete dynamics
# @show model.f(x,u) ≈ DoublePendulum.fc(x,u)
# @show fd(x,u) ≈ DoublePendulum.f(x,u,dt)
# @show f_aug(S) ≈ DoublePendulum.fc2(S)
# @show fd_aug(S) ≈ DoublePendulum.f2(S)
#
Df = S-> ForwardDiff.jacobian(fd_aug, S)
# @show Df(S) ≈ DoublePendulum.Df(S)

out = zeros(7)
f_aug! = iLQR.f_augmented!(model)
fd_aug! = iLQR.f_midpoint!(f_aug!)
f_aug!(out, S)
@show out ≈ f_aug(S)
fd_aug!(out,S)
@show out ≈ fd_aug(S)

const result = DiffResults.JacobianResult(out, S)
function f2_jacobian!(x::Array, u::Array, dt::Float64)
    ForwardDiff.jacobian!(result, fd_aug!, out, S)
    J = DiffResults.jacobian(result)
    A = J[1:n,1:n]
    B = J[1:n,n+1:n+m]
    return A,B
end

A1, B1 = DoublePendulum.f_jacobian(x0, u, dt)
A2, B2 = f2_jacobian!(x0, u, dt)
@show A1, A2
@show B1, B2
