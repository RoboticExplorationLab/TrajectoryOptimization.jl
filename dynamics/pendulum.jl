#
# ## Pendulum
# # https://github.com/HarvardAgileRoboticsLab/unscented-dynamic-programming/blob/master/pendulum_dynamics.m
# function pendulum_dynamics!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
#     m = 1.
#     l = 0.5
#     b = 0.1
#     lc = 0.5
#     I = 0.25
#     g = 9.81
#     ẋ[1] = x[2]
#     ẋ[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/I
# end
#
# n,m = 2,1
# pendulum = Model(pendulum_dynamics!,n,m) # inplace model
#
# # unknown mass
# function pendulum_dynamics_uncertain!(ẋ,x,u,w)
#     m = 1. + w[1]
#     l = 0.5
#     b = 0.1
#     lc = 0.5
#     I = 0.25
#     g = 9.81
#
#     ẋ[1] = x[2]
#     ẋ[2] = u[1]/(m*lc*lc) - g*sin(x[1])/lc - b*x[2]/(m*lc*lc)
#     return nothing
# end
#
# n = 2; m = 1; r = 1
# pendulum_uncertain = UncertainModel(pendulum_dynamics_uncertain!,n,m,r)


@with_kw mutable struct Pendulum{T} <: AbstractModel
    mass::T = 1.
    length::T = 0.5
    b::T = 0.1
    lc::T = 0.5
    I::T = 0.25
    g::T = 9.81
end

Base.size(::Pendulum) = 2,1

function dynamics(p::Pendulum, x, u)
    m = p.mass * p.lc * p.lc
    @SVector [x[2],
              u[1]/m - p.g*sin(x[1])/p.lc - p.b*x[2]/m]
end
@inline Base.position(::Pendulum, x) = @SVector zeros(3)
orientation(::Pendulum, x) = expm((pi-x[1])*@SVector [1,0,0.])
