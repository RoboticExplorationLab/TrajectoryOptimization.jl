
struct Satellite <: AbstractModel
    J::Diagonal{Float64,SVector{3,Float64}}
end

Satellite() = Satellite(Diagonal(@SVector ones(3)))

Base.size(::Satellite) = 7,3
Base.position(::Satellite, x::SVector) = @SVector zeros(3)
orientation(::Satellite, x::SVector) = @SVector [x[4], x[5], x[6], x[7]]

function dynamics(model::Satellite, x::SVector, u::SVector)
    ω = @SVector [x[1], x[2], x[3]]
    q = normalize(@SVector [x[4], x[5], x[6], x[7]])
    J = model.J

    ωdot = J\(u - ω × (J*ω))
    qdot = 0.5*Lmult(q)*Vmat()'ω
    return [ωdot; qdot]
end

function state_diff(model::Satellite, x::SVector, x0::SVector)::SVector{6}
    ω = @SVector [x[1], x[2], x[3]]
    q = @SVector [x[4], x[5], x[6], x[7]]
    ω0 = @SVector [x0[1], x0[2], x0[3]]
    q0 = @SVector [x0[4], x0[5], x0[6], x0[7]]

    δω = ω - ω0
    δq = Lmult(q0)'q
    ϕ = @SVector [δq[2]/δq[1], δq[3]/δq[1], δq[4]/δq[1]]
    return [δω; ϕ]
end

function state_diff_jacobian(model::Satellite, x::SVector)
    q = @SVector [x[4], x[5], x[6], x[7]]
    G = Lmult(q)*Vmat()'
    return @SMatrix [1 0 0 0 0 0;
                     0 1 0 0 0 0;
                     0 0 1 0 0 0;
                     0 0 0 G[1] G[5] G[ 9];
                     0 0 0 G[2] G[6] G[10];
                     0 0 0 G[3] G[7] G[11];
                     0 0 0 G[4] G[8] G[12];
                     ]
end

state_diff_size(::Satellite) = 6

TrajectoryOptimization.is_quat(::Satellite, z::KnotPoint) =
    (@SVector [0,0,0,1,1,1,1]), (@SVector [0,0,0,1,1,1])

function TrajectoryOptimization.state_diff_jacobian!(G, model::Satellite, Z::Traj)
    for k in eachindex(Z)
        G[k] = state_diff_jacobian(model, state(Z[k]))
    end
end



struct Satellite2{R} <: RigidBody{R}
    J::Diagonal{Float64,SVector{3,Float64}}
end

Satellite2() = Satellite2{UnitQuaternion{Float64,CayleyMap}}(Diagonal(@SVector ones(3)))

Base.size(::Satellite2) = 7,3
Base.position(::Satellite2, x::SVector) = @SVector zeros(3)
orientation(::Satellite2{UnitQuaternion{T,D}}, x::SVector{N,T2}) where {T,T2,D,N} =
    normalize(UnitQuaternion{T2,D}(x[4], x[5], x[6], x[7]))

function dynamics(model::Satellite2{R}, x::SVector, u::SVector) where R
    ω = @SVector [x[1], x[2], x[3]]
    q = orientation(model, x)
    J = model.J

    ωdot = J\(u - ω × (J*ω))
    qdot = kinematics(q,ω)
    return [ωdot; qdot]
end

function state_diff(model::Satellite2, x::SVector, x0::SVector)::SVector{6}
    ω = @SVector [x[1], x[2], x[3]]
    ω0 = @SVector [x0[1], x0[2], x0[3]]
    q = orientation(model, x)
    q0 = orientation(model, x0)

    δω = ω - ω0
    δq = q0\q
    ϕ = q ⊖ q0
    return [δω; ϕ]
end

function state_diff_jacobian(model::Satellite2, x::SVector)
    q = orientation(model, x)
    G = TrajectoryOptimization.∇differential(q)
    return @SMatrix [1 0 0 0 0 0;
                     0 1 0 0 0 0;
                     0 0 1 0 0 0;
                     0 0 0 G[1] G[5] G[ 9];
                     0 0 0 G[2] G[6] G[10];
                     0 0 0 G[3] G[7] G[11];
                     0 0 0 G[4] G[8] G[12];
                     ]
end

state_diff_size(::Satellite2) = 6

TrajectoryOptimization.is_quat(::Satellite2, z::KnotPoint) =
    (@SVector [0,0,0,1,1,1,1]), (@SVector [0,0,0,1,1,1])

function TrajectoryOptimization.state_diff_jacobian!(G, model::Satellite2, Z::Traj)
    for k in eachindex(Z)
        G[k] = state_diff_jacobian(model, state(Z[k]))
    end
end
