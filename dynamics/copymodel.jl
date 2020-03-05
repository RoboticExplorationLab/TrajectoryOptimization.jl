import TrajectoryOptimization: states
struct CopyModel{K,N,M,L} <: AbstractModel
    model::L
    ix::SVector{N,Int}
    iu::SVector{M,Int}
    xinds::Vector{SVector{N,Int}}
    uinds::Vector{SVector{M,Int}}
end

function CopyModel(model::L, K::Int) where L <: AbstractModel
    n,m = size(model)
    xind = SVector{n}(1:n)
    uind = SVector{m}(1:m)
    xinds = [xind .+ (i-1)*n for i = 1:K]
    uinds = [uind .+ (i-1)*m for i = 1:K]
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    return CopyModel{K,n,m,L}(model, ix, iu, xinds, uinds)
end

Base.size(::CopyModel{K,N,M}) where {K,N,M} = N*K,M*K

function build_state(model::CopyModel{K}, xs...) where K
    @assert length(xs) == K
    x = xs[1]
    for i = 2:K
        x = [x; xs[i]]
    end
    return x
end

function Base.zeros(model::CopyModel{K,N,M}) where {K,N,M}
    x,u = zeros(model.model)
    return repeat(x,K), repeat(u,K)
end

function Base.rand(model::CopyModel{K}) where K
    x,u = rand(model.model)
    for i = 2:K
        x_,u_ = rand(model.model)
        x = [x; x_]
        u = [u; u_]
    end
    return x,u
end


function states(model::CopyModel{K,N,M}, x) where {K,N,M}
    reshape(x,N,K)
    return [x[i] for i in model.xinds]
end

states(model::CopyModel{K,L}, z::KnotPoint, k::Int) where {L,K} =
    states(model, state(z), k)
function states(model::CopyModel{K,L}, x, k::Int) where {L,K}
    return x[model.xinds[k]]
end

function states(model::CopyModel, Z::Traj, k::Int)
    [state(z)[model.xinds[k]] for z in Z]
end

function controls(model::CopyModel{K,N,M}, u) where {K,N,M}
    reshape(u,M,K)
end

controls(model::CopyModel{K,L}, z::KnotPoint, k::Int) where {L,K} =
    controls(model, control(z), k)
function controls(model::CopyModel{K,L}, u, k::Int) where {L,K}
    return u[model.uinds[k]]
end

function controls(model::CopyModel, Z::Traj, k::Int)
    [control(z)[model.uinds[k]] for z in Z]
end

function TrajectoryOptimization.get_trajectory(model::CopyModel, Z::Traj, k::Int)
    xinds = model.xinds
    uinds = model.uinds
    map(Z) do z
        x = state(z)[xinds[k]]
        u = control(z)[uinds[k]]
        KnotPoint(x,u,z.dt,z.t)
    end
end

function TrajectoryOptimization.dynamics(model::CopyModel{K}, x, u, t=0.0) where K
    xinds = model.xinds
    uinds = model.uinds
    xdot = dynamics(model.model, x[xinds[1]], u[uinds[1]], t)
    for i = 2:K
        xdot_ = dynamics(model.model, x[xinds[i]], u[uinds[i]], t)
        xdot = [xdot; xdot_]
    end
    return xdot
end

function TrajectoryOptimization.discrete_dynamics(::Type{Q}, model::CopyModel{K},
        z::KnotPoint) where {K,Q <: TrajectoryOptimization.Implicit}
    xinds = model.xinds
    uinds = model.uinds
    x,u = state(z), control(z)
    xdot = discrete_dynamics(Q, model.model, x[xinds[1]], u[uinds[1]], z.t, z.dt)
    for i = 2:K
        xdot_ = discrete_dynamics(Q, model.model, x[xinds[i]], u[uinds[i]], z.t, z.dt)
        xdot = [xdot; xdot_]
    end
    return xdot
end

function TrajectoryOptimization.jacobian(model::CopyModel{K,N0,M0},
        z::KnotPoint{T,N,M,NM}) where {K,N0,M0,T,N,M,NM}
    A0 = @SMatrix zeros(N0,N0)
    B0 = @SMatrix zeros(N0,M0)
    ix = model.ix
    iu = model.iu

    xinds = model.xinds
    uinds = model.uinds
    x,u = state(z), control(z)

    # Process fist model
    z_ = [x[xinds[1]]; u[uinds[1]]]
    z_ = TrajectoryOptimization.StaticKnotPoint(z_, ix, iu, z.dt, z.t)
    ∇f = jacobian(model.model, z_)
    A,B = ∇f[ix,ix], ∇f[ix,iu]
    # append zeros after
    for i = 2:K
        A = [A  A0]
        B = [B  B0]
    end

    # loop over the rest of the models, appending below the first
    for i = 2:K
        z_ = [x[xinds[i]]; u[uinds[i]]]
        z_ = TrajectoryOptimization.StaticKnotPoint(z_, ix, iu, z.dt, z.t)
        ∇f = jacobian(model.model, z_)
        A_,B_ = ∇f[ix,ix], ∇f[ix,iu]

        # prepend zeros
        for j = 1:i-1
            A_ = [A0 A_]
            B_ = [B0 B_]
        end
        # append zeros
        for j = i+1:K
            A_ = [A_ A0]
            B_ = [B_ B0]
        end
        # stack with original
        A = [A; A_]
        B = [B; B_]
    end
    return [A B]
end

# function TrajectoryOptimization.discrete_jacobian(::Type{Q}, model::CopyModel{K,N0,M0},
#         z::KnotPoint{T,N,M,NM}) where {K,N0,M0,T,N,M,NM,Q<:TrajectoryOptimization.Implicit}
#     A0 = @SMatrix zeros(N0,N0)
#     B0 = @SMatrix zeros(N0,M0)
#     ix = model.ix
#     iu = model.iu
#     it = N0+M0+1
#
#     xinds = model.xinds
#     uinds = model.uinds
#     x,u = state(z), control(z)
#     dt = @SVector [z.dt]
#
#     z_ = [x[xinds[1]]; u[uinds[1]]; dt]
#     ∇f = discrete_jacobian(Q, model.model, z_, z.t, ix, iu)
#     A,B,C = ∇f[ix,ix], ∇f[ix,iu], ∇f[ix,it]
#     for i = 2:K
#         A = [A  A0]
#         B = [B  B0]
#     end
#     for i = 2:K
#         z_ = [x[xinds[i]]; u[uinds[i]]; dt]
#         ∇f = discrete_jacobian(Q, model.model, z_, z.t, ix, iu)
#         A_,B_,C_ = ∇f[ix,ix], ∇f[ix,iu], ∇f[ix,it]
#
#         for j = 1:i-1
#             A_ = [A0 A_]
#             B_ = [B0 B_]
#         end
#         for j = i+1:K
#             A_ = [A_ A0]
#             B_ = [B_ B0]
#         end
#         A = [A; A_]
#         B = [B; B_]
#         C = [C; C_]
#     end
#     return [A B C]
# end

function discrete_jacobian!(::Type{Q}, ∇f, model::Dynamics.CopyModel{K,N0,M0},
		z::KnotPoint{T,N,M,NM}) where {T,N,M,NM,K,N0,M0,Q<:TrajectoryOptimization.Implicit}
    xinds = model.xinds
    uinds = model.uinds
	ix = model.ix
	iu = model.iu
	iz = [ix; iu; @SVector [N0+M0+1]]
	x,u = state(z), control(z)
	for i = 1:K
		z_ = [x[xinds[i]]; u[uinds[i]]]
		z_ = StaticKnotPoint(z_, ix, iu, z.dt, z.t)
		∇f_ = discrete_jacobian(Q, model.model, z_)
		A,B,C = ∇f_[ix,ix], ∇f_[ix,iu], ∇f_[ix,N0+M0+1]

		ix_ = ix .+ (i-1)*N0
		iu_ = iu .+ (i-1)*M0 .+ N .- N0
		it_ = NM+1
		∇f[ix_,ix_] .= A
		∇f[ix_,iu_] .= B
		∇f[ix_,it_] .= C
	end
end

function state_diff_size(model::CopyModel{K}) where K
    return state_diff_size(model.model)*K
end

state_diff(model::CopyModel, x::SVector, x0::SVector) = x-x0
function state_diff(model::CopyModel{K,N,M,L}, x::SVector, x0::SVector) where {K,N,M,L<:RigidBody}
    xinds = model.xinds
    dx = state_diff(model.model, x[xinds[1]], x0[xinds[1]])
    for i = 2:K
        dx_ = state_diff(model.model, x[xinds[i]], x0[xinds[i]])
        dx = [dx; dx_]
    end
    return dx
end

TrajectoryOptimization.state_diff_jacobian!(G, model::CopyModel, Z::Traj) = nothing
function TrajectoryOptimization.state_diff_jacobian!(G,
        model::CopyModel{<:Any,<:Any,<:Any,L}, Z::Traj) where {L<:RigidBody{R}} where R
    for k in eachindex(Z)
        G[k] .= state_diff_jacobian(model, state(Z[k]))
    end
end


state_diff_jacobian(::CopyModel, x::SVector) = I
@generated function state_diff_jacobian(model::CopyModel{K,N,M,L},
        x::SVector) where {K,N,M,L<:RigidBody{R}} where R
    if R <: UnitQuaternion
        if R <: UnitQuaternion{T,IdentityMap} where T
            return :(I)
        else
            G0 = @SMatrix zeros(N,N-1)
        end
    else
        G0 = @SMatrix zeros(N,N)
    end
    quote
        xinds = model.xinds
        G = state_diff_jacobian(model.model, x[xinds[1]])
        for i = 2:K
            G = [G $G0]
        end
        for i = 2:K
            G_ = state_diff_jacobian(model.model, x[xinds[i]])
            for j = 1:i-1
                G_ = [$G0 G_]
            end
            for j = i+1:K
                G_ = [G_ $G0]
            end
            G = [G; G_]
        end
        G
    end
end

@generated function TrajectoryOptimization.∇²differential(model::CopyModel{K,N,M,L},
        x::SVector, dx::SVector) where {K,N,M,L<:RigidBody{R}} where R

    if R <: UnitQuaternion
        if R <: UnitQuaternion{T,IdentityMap} where T
            return :(I*0)
        else
            G0 = @SMatrix zeros(N-1,N-1)
        end
    else
        G0 = @SMatrix zeros(N,N)
    end
    quote
        dx_ = reshape(dx,:,K)
        ix_ = SVector{12}(1:12)
        xinds = model.xinds
        G = TrajectoryOptimization.∇²differential(model.model, x[xinds[1]], dx[ix_])
        for i = 2:K
            G = [G $G0]
        end
        for i = 2:K
            dx_ = dx[ix_ .+ (i-1)*12]
            G_ = TrajectoryOptimization.∇²differential(model.model, x[xinds[i]], dx_)
            for j = 1:i-1
                G_ = [$G0 G_]
            end
            for j = i+1:K
                G_ = [G_ $G0]
            end
            G = [G; G_]
        end
        G
    end
end

function ∇²differential!(G, model::Dynamics.CopyModel{K,N,M,L},
        x::SVector, dx::Vector) where {K,N,M,L<:RigidBody{R}} where R
    ix = SVector{12}(1:12)
    xinds = model.xinds
    for i = 1:K
        ix_ = ix .+ (i-1)*12
        dx_ = dx[ix_]
        G_ = ∇²differential(model.model, x[xinds[i]], dx_)
        G[ix_, ix_] .= G_
    end
end
