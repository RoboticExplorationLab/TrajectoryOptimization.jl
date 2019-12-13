# ## Cartpole
# import TrajectoryOptimization
# traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
# urdf_folder = joinpath(traj_folder, "dynamics/urdf")
# urdf_cartpole = joinpath(urdf_folder, "cartpole.urdf")
#
# cartpole_urdf = Model(urdf_cartpole,[1.;0.]) # underactuated, only control of slider
#
# function cartpole_dynamics!(ẋ::AbstractVector{T}, x::AbstractVector{T}, u::AbstractVector{T}) where T
#     mc = 1.0  # mass of the cart in kg (10)
#     mp = 0.2    # mass of the pole (point mass at the end) in kg
#     l = 0.5   # length of the pole in m
#     g = 9.81  # gravity m/s^2
#
#     q = x[1:2]
#     qd = x[3:4]
#
#     if isfinite(q[2])
#         s = sin(q[2])
#         c = cos(q[2])
#     else
#         s = Inf
#         c = Inf
#     end
#
#     H = [mc+mp mp*l*c; mp*l*c mp*l^2]
#     C = [0 -mp*qd[2]*l*s; 0 0]
#     G = [0; mp*g*l*s]
#     B = [1; 0]
#
#     qdd = -H\(C*qd + G - B*u[1])
#
#     ẋ[1:2] = qd
#     ẋ[3:4] = qdd
#     return nothing
# end

# n,m = 4,1
# cartpole = Model(cartpole_dynamics!,n,m)
#
# # unknown friction
# function cartpole_dynamics_uncertain!(ẋ, x, u, w)
#     mc = 1.0  # mass of the cart in kg (10)
#     mp = 0.2    # mass of the pole (point mass at the end) in kg
#     l = 0.5   # length of the pole in m
#     g = 9.81  # gravity m/s^2
#     mu = 0.1
#
#     q = x[1:2]
#     qd = x[3:4]
#
#     if isfinite(q[2])
#         s = sin(q[2])
#         c = cos(q[2])
#     else
#         s = Inf
#         c = Inf
#     end
#
#     H = [mc+mp mp*l*c; mp*l*c mp*l^2]
#     C = [0 -mp*qd[2]*l*s; 0 0]
#     G = [0; mp*g*l*s]
#     B = [1; 0]
#
#     qdd = -H\(C*qd + G - B*u[1] - B*w[1])
#
#     ẋ[1:2] = qd
#     ẋ[3:4] = qdd
#
#     return nothing
# end
#
# n = 4; m = 1; r = 1
# cartpole_uncertain = UncertainModel(cartpole_dynamics_uncertain!,n,m,r)

struct Cartpole{T} <: AbstractModel
    n::Int
    m::Int
    mc::T
    mp::T
    l::T
    g::T
end

Cartpole() = Cartpole(4,1,
    1.0, 0.2, 0.5, 9.81)

function dynamics(model::Cartpole, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end
