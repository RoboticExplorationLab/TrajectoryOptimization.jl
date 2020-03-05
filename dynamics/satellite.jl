
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

function TrajectoryOptimization.state_diff_jacobian!(G, model::Satellite, Z::Traj)
    for k in eachindex(Z)
        G[k] = state_diff_jacobian(model, state(Z[k]))
    end
end



struct Satellite2{R,B} <: RigidBody{R}
    J::Diagonal{Float64,SVector{3,Float64}}
end

Satellite2() = Satellite2{UnitQuaternion{Float64,CayleyMap},true}(Diagonal(@SVector ones(3)))
Satellite2{R}(;use_rot=true) where R= Satellite2{R,use_rot}(Diagonal(@SVector ones(3)))

Base.size(::Satellite2{<:UnitQuaternion}) = 7,3
Base.size(::Satellite2) = 6,3
Base.position(::Satellite2, x::SVector, unitnorm=true) = @SVector zeros(3)
function orientation(::Satellite2{UnitQuaternion{T,D}}, x::SVector{N,T2},
        unitnorm=true) where {T,T2,D,N}
    q = UnitQuaternion{T2,D}(x[4], x[5], x[6], x[7])
    if unitnorm
        return normalize(q)
    end
    q
end
orientation(::Satellite2{R}, x::SVector{N,T2}, unitnorm=false) where {R,N,T2} =
    R(T2,x[4],x[5],x[6])

function build_state(model::Satellite2{R}, ω, q) where R
    q = SVector(R(q))
    return [ω; q]
end

function Base.rand(::Satellite2{R}) where R
    u = @SVector rand(3)
    ω = @SVector rand(3)
    q = rand(R)
    return [ω; SVector(q)], u
end

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
    ϕ = q ⊖ q0
    return [δω; ϕ]
end

function state_diff(model::Satellite2{R,false}, x::SVector, x0::SVector) where R
    return x - x0
end


function state_diff_jacobian(model::Satellite2{<:UnitQuaternion}, x::SVector)
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

function state_diff_jacobian(model::Satellite2, x::SVector)
    q = orientation(model, x)
    G = TrajectoryOptimization.∇differential(q)
    return @SMatrix [1 0 0 0 0 0;
                     0 1 0 0 0 0;
                     0 0 1 0 0 0;
                     0 0 0 G[1] G[4] G[7];
                     0 0 0 G[2] G[5] G[8];
                     0 0 0 G[3] G[6] G[9];
                     ]
end

function state_diff_jacobian(model::Satellite2{R,false}, x::SVector) where R
    return I
end


import TrajectoryOptimization: ∇²differential
function ∇²differential(model::Satellite2, x::SVector, dx::SVector)
    q = orientation(model, x)
    dq = SVector(orientation(model, dx, false))  # gradient wrt the rotation
    G2 = ∇²differential(q, dq)
    return @SMatrix [0 0 0 0 0 0;
                     0 0 0 0 0 0;
                     0 0 0 0 0 0;
                     0 0 0 G2[1] G2[4] G2[7];
                     0 0 0 G2[2] G2[5] G2[8];
                     0 0 0 G2[3] G2[6] G2[9];
                     ]
end

function ∇²differential(model::Satellite2{R,false}, x::SVector, dx::SVector) where R
    return I*0
end

state_diff_size(::Satellite2) = 6
state_diff_size(::Satellite2{<:UnitQuaternion,false}) = 7


function TrajectoryOptimization.state_diff_jacobian!(G, model::Satellite2, Z::Traj)
    for k in eachindex(Z)
        G[k] = state_diff_jacobian(model, state(Z[k]))
    end
end
