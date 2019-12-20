import TrajectoryOptimization.Dynamics: forces, moments, inertia, inertia_inv, mass_matrix

function dynamics(model::RigidBody{<:Quat}, x, u)
    q = normalize(Quaternion(x[4],x[5],x[6],x[7]))
    v = SVector{3}(x[8],x[9],x[10])
    ω = SVector{3}(x[11],x[12],x[13])
    F = forces(model, x, u)
    τ = moments(model, x, u)
    M = mass_matrix(model, x, u)
    J = inertia(model, x, u)
    Jinv = inertia_inv(model, x, u)

    xdot = v
    qdot = 0.5*q*Quaternion(0,ω[1],ω[2],ω[3])
    vdot = M\F
    ωdot = Jinv*(τ - ω × (J*ω))
    @SVector [xdot[1], xdot[2], xdot[3],
              qdot.s,  qdot.v1, qdot.v2, qdot.v3,
              vdot[1], vdot[2], vdot[3],
              ωdot[1], ωdot[2], ωdot[3]]
end

function dynamics(model::RigidBody{MRP}, x, u)
    p = SVector{3}(x[4],x[5],x[6])
    v = SVector{3}(x[7],x[8],x[9])
    ω = SVector{3}(x[10],x[11],x[12])
    F = forces(model, x, u)
    M = moments(model, x, u)
    J = inertia(model, x, u)
    Jinv = inertia_inv(model, x, u)

    xdot = v
    pdot = 0.25*MRP_kinematics(p)*ω
    vdot = MRP_rotate_vec(p,F)
    ωdot = Jinv*(M - ω × J*ω)
    @SVector [xdot[1], xdot[2], xdot[3],
              pdot[1], pdot[2], pdot[3],
              vdot[1], vdot[2], vdot[3],
              ωdot[1], ωdot[2], ωdot[3]]
end


@inline mass_matrix(::RigidBody, x, u) = throw(ErrorException("Not Implemented"))
@inline forces(::RigidBody, x, u)::SVector{3} = throw(ErrorException("Not implemented"))
@inline moments(::RigidBody, x, u)::SVector{3} = throw(ErrorException("Not implemented"))
@inline inertia(::RigidBody, x, u)::SMatrix{3,3} = throw(ErrorException("Not implemented"))
@inline inertia_inv(::RigidBody, x, u)::SMatrix{3,3} = throw(ErrorException("Not implemented"))

function state_diff(::RigidBody{<:Quat{D}}, x::SVector{N,T}, x0::SVector{N,T}) where {N,T,D}
    q  = Quaternion(x[4],x[5],x[6],x[7])
    q0 = Quaternion(x0[4],x0[5],x0[6],x0[7])
    # q = x[inds]
    # q0 = x0[inds]
    # δq = quat_diff(q, q0)
    δq = inv(q0)*q
    δr = differential_rotation(D,δq)::SVector{3,T}
    δx = x - x0
    δx = @SVector [δx[1], δx[2], δx[3],
                   δr[1], δr[2], δr[3],
                   δx[8], δx[9], δx[10],
                   δx[11], δx[12], δx[13]]
end

function state_diff_jacobian(::RigidBody, x0::SVector{N,T}) where {N,T}
    q0  = Quaternion(x0[4],x0[5],x0[6],x0[7])
    G = quat_diff_jacobian(q0)
    return
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

@inline state_diff_size(::RigidBody) = 12
