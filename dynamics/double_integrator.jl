# function double_integrator_dynamics!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
#     ẋ[1] = x[2]
#     ẋ[2] = u[1]
# end
#
# n = 2
# m = 1
#
# doubleintegrator = Model(double_integrator_dynamics!,n,m)

struct DoubleIntegrator{N,M} <: AbstractModel
    pos::SVector{M,Int}
    vel::SVector{M,Int}
end

function DoubleIntegrator(D=1)
    pos = SVector{D,Int}(1:D)
    vel = SVector{D,Int}(D .+ (1:D))
    DoubleIntegrator{2D,D}(pos,vel)
end

Base.size(::DoubleIntegrator{N,M}) where {N,M} = N,M


@generated function dynamics(di::DoubleIntegrator{N,M}, x, u) where {N,M}
    vel = [:(x[$i]) for i = M+1:N]
    us = [:(u[$i]) for i = 1:M]
    :(SVector{$N}($(vel...),$(us...)))
end
