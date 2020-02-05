using Distributions

struct NoisyRB{L,D,W} <: RigidBody{D}
    model::L
    noise::W
end

@inline Base.size(model::NoisyRB) = size(model.model)

function NoisyRB(model::L, noise::W) where {L<:RigidBody{D}, W<:Distribution{Multivariate,Continuous}} where D
    NoisyRB{L,D,W}(model, noise)
end

function dynamics(model::NoisyRB, x::SVector, u::SVector)
    r,q,v,ω = parse_state(model.model, x)

    # Get noise
    w = rand(model.noise)
    F_ = @SVector [w[1], w[2], w[3]]
    τ_ = @SVector [w[4], w[5], w[6]]

    # Original dynamics
    F,τ = wrenches(model.model, x, u)
    M = mass_matrix(model.model, x, u)
    J = inertia(model.model, x, u)
    Jinv = inertia_inv(model.model, x, u)

    # Add noise wrench
    F += F_
    τ += τ_

    xdot = v
    qdot = kinematics(q,ω)
    vdot = M\F
    ωdot = Jinv*(τ - ω × (J*ω))

    build_state(model, xdot, qdot, vdot, ωdot)
end

@inline state_diff_size(model::NoisyRB) = state_diff_size(model.model)

@inline state_diff(model::NoisyRB, x::SVector, x0::SVector) =
    state_diff(model.model, x, x0)

@inline state_diff_jacobian(model::NoisyRB, x::SVector) =
    state_diff_jacobian(model.model, x)

@inline TrajectoryOptimization.∇²differential(model::NoisyRB, x::SVector, dx::SVector) =
    TrajectoryOptimization.∇²differential(model.model, x, dx)

@inline TrajectoryOptimization.inverse_map_jacobian(model::NoisyRB, x::SVector) =
    TrajectoryOptimization.inverse_map_jacobian(model.model, x)

@inline TrajectoryOptimization.inverse_map_∇jacobian(model::NoisyRB, x::SVector, b::SVector) =
    TrajectoryOptimization.inverse_map_∇jacobian(model.model, x, b)
