
############################################################################################
#                               INFEASIBLE MODELS                                          #
############################################################################################
""" $(TYPEDEF)
An infeasible model is an augmented dynamics model that makes the system artifically fully
actuated by augmenting the control vector with `n` additional controls. The dynamics are
handled explicitly in discrete time:

``x_{k+1} = f(x_k,u_k,dt) + w_k``

where ``w_k`` are the additional `n`-dimensional controls. In practice, these are constrained
to be zero by the end of the solve.

# Constructors
```julia
InfeasibleModel(model::AbstractModel)
```
"""
struct InfeasibleModel{N,M,D<:AbstractModel} <: AbstractModel
    model::D
    _u::SVector{M,Int}  # inds to original controls
    _ui::SVector{N,Int} # inds to infeasible controls
end

function InfeasibleModel(model::AbstractModel)
    n,m = size(model)
    _u  = SVector{m}(1:m)
    _ui = SVector{n}((1:n) .+ m)
    InfeasibleModel(model, _u, _ui)
end

function Base.size(model::InfeasibleModel)
    n,m = size(model.model)
    return n, n+m
end

Dynamics.dynamics(::InfeasibleModel, x, u) =
    throw(ErrorException("Cannot evaluate continuous dynamics on an infeasible model"))

@generated function Dynamics.discrete_dynamics(::Type{Q}, model::InfeasibleModel{N,M},
        z::KnotPoint{T,N}) where {T,N,M,Q<:Implicit}
    _u = SVector{M}((1:M) .+ N)
    _ui = SVector{N}((1:N) .+ (N+M))
    quote
        x = state(z)
        dt = z.dt
        u0 = z.z[$_u]
        ui = z.z[$_ui]
        discrete_dynamics($Q, model.model, x, u0, z.t, dt) + ui
    end
end

# @generated function discrete_jacobian(::Type{Q}, model::InfeasibleModel{N,M},
#         z::KnotPoint{T,N,NM,L}) where {T,N,M,NM,L,Q<:Implicit}
#
#     ∇ui = [(@SMatrix zeros(N,N+M)) Diagonal(@SVector ones(N)) @SVector zeros(N)]
#     _x = SVector{N}(1:N)
#     _u = SVector{M}((1:M) .+ N)
#     _z = SVector{N+M}(1:N+M)
#     _ui = SVector{N}((1:N) .+ (N+M))
#     zi = [:(z.z[$i]) for i = 1:N+M]
#     NM1 = N+M+1
#     ∇u0 = @SMatrix zeros(N,N)
#
#     quote
#         # Build KnotPoint for original model
#         s0 = SVector{$NM1}($(zi...), z.dt)
#
#         u0 = z.z[$_u]
#         ui = z.z[$_ui]
#         ∇f = discrete_jacobian($Q, model.model, s0, z.t, $_x, $_u)::SMatrix{N,NM+1}
#         ∇dt = ∇f[$_x, N+M+1]
#         [∇f[$_x, $_z] $∇u0 ∇dt] + $∇ui
#     end
# end

@inline Dynamics.rotation_type(model::InfeasibleModel) where D = rotation_type(model.model)

@generated function Dynamics.discrete_jacobian!(::Type{Q}, ∇f, model::InfeasibleModel{N,M},
        z::KnotPoint{T,N,NM,L}) where {T,N,M,NM,L,Q<:Implicit}

    ∇ui = [(@SMatrix zeros(N,N+M)) Diagonal(@SVector ones(N)) @SVector zeros(N)]
    _x = SVector{N}(1:N)
    _u = SVector{M}((1:M) .+ N)
    _z = SVector{N+M}(1:N+M)
    _ui = SVector{N}((1:N) .+ (N+M))
    zi = [:(z.z[$i]) for i = 1:N+M]
    NM1 = N+M+1
    ∇u0 = @SMatrix zeros(N,N)

    quote
        # Build KnotPoint for original model
        s0 = SVector{$NM1}($(zi...), z.dt)

        u0 = z.z[$_u]
        ui = z.z[$_ui]
		z_ = StaticKnotPoint(z.z[$_z], $_x, $_u, z.dt, z.t)
		∇f_ = uview(∇f, 1:N, 1:$NM1)
        discrete_jacobian!($Q, ∇f_, model.model, z_)
		∇f[$_x, N+NM+1] .= ∇f_[$_x, N+M+1] # ∇dt
		∇f[$_x, $_ui] .= Diagonal(@SVector ones(N))
		return
		# ∇f[$_x,$_ui]
        # [∇f[$_x, $_z] $∇u0 ∇dt] + $∇ui
    end
end

function Dynamics.state_diff(model::InfeasibleModel, x::SVector, x0::SVector)
	state_diff(model.model, x, x0)
end

# function state_diff_jacobian(model::InfeasibleModel, x::SVector)
# 	state_diff_jacobian(model.model, x)
# end

function Dynamics.state_diff_jacobian!(G, model::InfeasibleModel, Z::Traj)
	state_diff_jacobian!(G, model.model, Z)
end

function Dynamics.∇²differential(model::InfeasibleModel, x::SVector, dx::SVector)
	return ∇²differential(model.model, x, dx)
end

Dynamics.state_diff_size(model::InfeasibleModel) = state_diff_size(model.model)

Base.position(model::InfeasibleModel, x::SVector) = position(model.model, x)

Dynamics.orientation(model::InfeasibleModel, x::SVector) = orientation(model.model, x)

"Calculate a dynamically feasible initial trajectory for an infeasible problem, given a
desired trajectory"
function infeasible_trajectory(model::InfeasibleModel{n,m}, Z0::Vector{<:KnotPoint{T,n,m}}) where {T,n,m}
    x,u = zeros(model)
    ui = @SVector zeros(n)
    Z = [KnotPoint(state(z), [control(z); ui], z.dt, z.t) for z in Z0]
    N = length(Z0)
    for k = 1:N-1
        Dynamics.propagate_dynamics(RK3, model, Z[k+1], Z[k])
        x′ = state(Z[k+1])
        u_slack = state(Z0[k+1]) - x′
        u = [control(Z0[k]); u_slack]
        Dynamics.set_control!(Z[k], u)
        Dynamics.set_state!(Z[k+1], x′ + u_slack)
    end
    return Z
end
