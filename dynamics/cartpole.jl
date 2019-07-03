## Cartpole
import TrajectoryOptimization
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf_cartpole = joinpath(urdf_folder, "cartpole.urdf")

cartpole_model_urdf = Model(urdf_cartpole,[1.;0.]) # underactuated, only control of slider

function cartpole_dynamics!(ẋ::AbstractVector{T}, x::AbstractVector{T}, u::AbstractVector{T}) where T
    mc = 10.0  # mass of the cart in kg (10)
    mp = 1.0    # mass of the pole (point mass at the end) in kg
    l = 0.5   # length of the pole in m
    g = 9.81  # gravity m/s^2

    q = x[1:2]
    qd = x[3:4]

    if isfinite(q[2])
        s = sin(q[2])
        c = cos(q[2])
    else
        s = Inf
        c = Inf
    end

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0; mp*g*l*s]
    B = [1; 0]

    qdd = -H\(C*qd + G - B*U')

    ẋ[1:2] = qd
    ẋ[3:4] = qdd
    return nothing
end

n,m = 4,1
cartpole_model = Model(cartpole_dynamics!,n,m)

function cartpole_dynamics_uncertain!(ẋ, x, u, w)
    mc = 1.0  # mass of the cart in kg (10)
    mp = 0.2    # mass of the pole (point mass at the end) in kg
    l = 0.5   # length of the pole in m
    g = 9.81  # gravity m/s^2
    mu = 0.1

    q = x[1:2]
    qd = x[3:4]

    if isfinite(q[2])
        s = sin(q[2])
        c = cos(q[2])
    else
        s = Inf
        c = Inf
    end

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0; mp*g*l*s]
    B = [1; 0]

    qdd = -H\(C*qd + G - B*u[1] - B*w[1])

    ẋ[1:2] = qd
    ẋ[3:4] = qdd

    return nothing
end

n = 4; m = 1; r = 1
cartpole_model_uncertain = UncertainModel(cartpole_dynamics_uncertain!,n,m,r)
# model = UncertainModel(cartpole_dynamics_uncertain!,n,m,r)
#
#
# # Continuous dynamics (uncertain)
# function fc(z)
#     ż = zeros(eltype(z),n)
#     model.f(ż,z[1:n],z[n .+ (1:m)],z[(n+m) .+ (1:r)])
#     return ż
# end
#
# function fc(x,u,w)
#     ẋ = zero(x)
#     model.f(ẋ,x,u,w)
#     return ẋ
# end
#
# using ForwardDiff
# ∇fc(z) = ForwardDiff.jacobian(fc,z)
# ∇fc(x,u,w) = ∇fc([x;u;w])
# dfcdx(x,u,w) = ∇fc(x,u,w)[:,1:n]
# dfcdu(x,u,w) = ∇fc(x,u,w)[:,n .+ (1:m)]
# dfcdw(x,u,w) = ∇fc(x,u,w)[:,(n+m) .+ (1:r)]
#
# xm(y,x,u,h) = 0.5*(y + x) + h/8*(fc(x,u,zeros(r)) - fc(y,u,zeros(r)))
# dxmdy(y,x,u,h) = 0.5*I - h/8*dfcdx(y,u,zeros(r))
# dxmdx(y,x,u,h) = 0.5*I + h/8*dfcdx(x,u,zeros(r))
# dxmdu(y,x,u,h) = h/8*(dfcdu(x,u,zeros(r)) - dfcdu(y,u,zeros(r)))
# dxmdh(y,x,u,h) = 1/8*(fc(x,u,zeros(r)) - fc(y,u,zeros(r)))
#
# # cubic interpolation on state
# F(y,x,u,h) = y - x - h/6*fc(x,u,zeros(r)) - 4*h/6*fc(xm(y,x,u,h),u,zeros(r)) - h/6*fc(y,u,zeros(r))
# F(z) = F(z[1:n],z[n .+ (1:n)],z[(n+n) .+ (1:m)],z[n+n+m+1])
# ∇F(Z) = ForwardDiff.jacobian(F,Z)
# dFdy(y,x,u,h) = I - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdy(y,x,u,h) - h/6*dfcdx(y,u,zeros(r))
# dFdx(y,x,u,h) = -I - h/6*dfcdx(x,u,zeros(r)) - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdx(y,x,u,h)
# dFdu(y,x,u,h) = -h/6*dfcdu(x,u,zeros(r)) - 4*h/6*(dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdu(y,x,u,h) + dfcdu(xm(y,x,u,h),u,zeros(r))) - h/6*dfcdu(y,u,zeros(r))
# dFdh(y,x,u,h) = -1/6*fc(x,u,zeros(r)) - 4/6*fc(xm(y,x,u,h),u,zeros(r)) - 4*h/6*dfcdx(xm(y,x,u,h),u,zeros(r))*dxmdh(y,x,u,h) - 1/6*fc(y,u,zeros(r))
# dFdw(y,x,u,h) = -h/6*dfcdw(x,u,zeros(r)) - 4*h/6*dfcdw(xm(y,x,u,h),u,zeros(r)) - h/6*dfcdw(y,u,zeros(r))
#
# cartpole_dynamics_uncertain!(rand(n),rand(n),rand(m),rand(r))
# fc([rand(n);rand(m);rand(r)])
# ∇fc([rand(n);rand(m);rand(r)])
# ∇fc(rand(n),rand(m),rand(r))
# dfcdx(rand(n),rand(m),rand(r))
# dfcdu(rand(n),rand(m),rand(r))
# dfcdw(rand(n),rand(m),rand(r))
#
# F(rand(n),rand(n),rand(m),1.0)
# dFdy(rand(n),rand(n),rand(m),1.0)
# dFdx(rand(n),rand(n),rand(m),1.0)
# dFdu(rand(n),rand(n),rand(m),1.0)
# dFdh(X[2],X[1],U[1],dt)
# -dFdy(rand(n),rand(n),rand(m),1.0)\dFdx(rand(n),rand(n),rand(m),1.0)
