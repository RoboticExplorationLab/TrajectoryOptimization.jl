import TrajectoryOptimization.Dynamics: forces, moments, inertia, inertia_inv, mass_matrix
import TrajectoryOptimization.Rotation

function Base.rand(model::RigidBody{D}) where {D}
    n,m = size(model)
    r = @SVector rand(3)
    q = rand(D)
    v = @SVector rand(3)
    ω = @SVector rand(3)
    x = build_state(model, r, SVector(q), v, ω)
    u = @SVector rand(m)  # NOTE: this is type unstable
    return x,u
end

@inline rotation_type(::RigidBody{D}) where D = D

@inline Base.position(model::RigidBody, x) = SVector{3}(x[1],x[2],x[3])
@inline orientation(model::RigidBody, x) = SVector{3}(x[4],x[5],x[6])
@inline linear_velocity(model::RigidBody, x) = SVector{3}(x[7],x[8],x[9])
@inline angular_velocity(model::RigidBody, x) = SVector{3}(x[10],x[11],x[12])

@inline orientation(model::RigidBody{UnitQuaternion{T,D}}, x::SVector{N,T2}) where {T,D,N,T2} =
    normalize(UnitQuaternion{T2,D}(x[4],x[5],x[6],x[7]))
@inline linear_velocity(model::RigidBody{<:UnitQuaternion}, x) = SVector{3}(x[8],x[9],x[10])
@inline angular_velocity(model::RigidBody{<:UnitQuaternion}, x) = SVector{3}(x[11],x[12],x[13])

function parse_state(model::RigidBody, x)
    r = position(model, x)
    p = orientation(model, x)
    v = linear_velocity(model, x)
    ω = angular_velocity(model, x)
    return r, p, v, ω
end

function build_state(model::RigidBody{<:UnitQuaternion}, x, q, v, ω)
    @SVector [x[1], x[2], x[3],
              q[1], q[2], q[3], q[4],
              v[1], v[2], v[3],
              ω[1], ω[2], ω[3]]
end

function build_state(model::RigidBody{<:Rotation}, x, q, v, ω)
    @SVector [x[1], x[2], x[3],
              q[1], q[2], q[3],
              v[1], v[2], v[3],
              ω[1], ω[2], ω[3]]
end

function dynamics(model::RigidBody{D}, x, u) where D

    r,q,v,ω = parse_state(model, x)

    F = forces(model, x, u)
    τ = moments(model, x, u)
    M = mass_matrix(model, x, u)
    J = inertia(model, x, u)
    Jinv = inertia_inv(model, x, u)

    xdot = v
    qdot = kinematics(q,ω)
    vdot = M\F
    ωdot = Jinv*(τ - ω × (J*ω))

    build_state(model, xdot, qdot, vdot, ωdot)
end

@inline mass_matrix(::RigidBody, x, u) = throw(ErrorException("Not Implemented"))
@inline forces(::RigidBody, x, u)::SVector{3} = throw(ErrorException("Not implemented"))
@inline moments(::RigidBody, x, u)::SVector{3} = throw(ErrorException("Not implemented"))
@inline inertia(::RigidBody, x, u)::SMatrix{3,3} = throw(ErrorException("Not implemented"))
@inline inertia_inv(::RigidBody, x, u)::SMatrix{3,3} = throw(ErrorException("Not implemented"))

function state_diff(::RigidBody{<:Quat{D}}, x::SVector{N,T}, x0::SVector{N,T}) where {N,T,D}
    r,q,v,ω = parse_state(model, x)
    r0,q0,v0,ω0 = parse_state(model, x0)
    δr = r - r0
    δq = q ⊖ q0
    δv = v - v0
    δω = ω - ω0
    build_state(model, δr, δq, δv, δω)
end

function state_diff_jacobian(::RigidBody{<:UnitQuaternion}, x0::SVector{N,T}) where {N,T}
    q0 = orientation(model, x0)
    G = ∇differential(q0)
    I1 = @SMatrix [1 0 0 0 0 0 0 0 0 0 0 0 0;
                   0 1 0 0 0 0 0 0 0 0 0 0 0;
                   0 0 1 0 0 0 0 0 0 0 0 0 0;
                   0 0 0 G[1] G[4] G[7] G[10] 0 0 0 0 0 0;
                   0 0 0 G[2] G[5] G[8] G[11] 0 0 0 0 0 0;
                   0 0 0 G[3] G[6] G[9] G[12] 0 0 0 0 0 0;
                   0 0 0 0 0 0 0 1 0 0 0 0 0;
                   0 0 0 0 0 0 0 0 1 0 0 0 0;
                   0 0 0 0 0 0 0 0 0 1 0 0 0;
                   0 0 0 0 0 0 0 0 0 0 1 0 0;
                   0 0 0 0 0 0 0 0 0 0 0 1 0;
                   0 0 0 0 0 0 0 0 0 0 0 0 1.]
end

function state_diff_jacobian(::RigidBody{<:Rotation}, x0::SVector{N,T}) where {N,T}
    I1 = Diagonal(@SVector ones(N))
end

@inline state_diff_size(::RigidBody) = 12
