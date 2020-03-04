@with_kw struct FreeBody{R,T} <: RigidBody{R}
    mass::T = 1.0
    J::Diagonal{T,SVector{3,T}} = Diagonal(@SVector ones(3))
    Jinv::Diagonal{T,SVector{3,T}} = Diagonal(@SVector ones(3))
end

(::FreeBody)(;kwargs...) = FreeBody{UnitQuaternion{Float64,CayleyMap},Float64}(;kwargs...)

function wrenches(model::FreeBody, x::SVector, u::SVector)
    F = forces(model, x, u)
    M = moments(model, x, u)
    return F,M
end
function forces(model::FreeBody, x::SVector, u::SVector)
    q = orientation(model, x)
    F = @SVector [u[1], u[2], u[3]]
    q*F  # world frame
end

function moments(model::FreeBody, x::SVector, u::SVector)
    return @SVector [u[4], u[5], u[6]]  # body frame
end

Base.size(::FreeBody{<:UnitQuaternion}) = 13,6
Base.size(::FreeBody{<:Rotation}) = 12,6

inertia(model::FreeBody, x, u) = model.J
inertia_inv(model::FreeBody, x, u) = model.Jinv
mass_matrix(model::FreeBody, x, u) = Diagonal(@SVector fill(model.mass,3))
