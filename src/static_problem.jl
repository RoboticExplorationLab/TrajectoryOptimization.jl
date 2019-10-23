using StaticArrays

mutable struct KnotPoint{T,N,M,NM}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    _inds::SVector{NM,Bool}
    dt::T
end

function KnotPoint(x::AbstractVector, u::AbstractVector, dt::Float64)
    n = length(x)
    m = length(u)
    xinds = ones(Bool, n+m)
    xinds[n+1:end] .= 0
    _x = SVector{n}(1:n)
    _u = SVector{m}(n .+ (1:m))
    _inds = SVector{n+m}(xinds)
    z = SVector{n+m}([x;u])
    KnotPoint(z, _x, _u, _inds, dt)
end

# function KnotPoint(x::Vector, u::Vector, dt::Float64)
#     n = length(x)
#     m = length(u)
#     xinds = ones(Bool, n+m)
#     xinds[n+1:end] .= 0
#     _x = SVector{n}(1:n)
#     _u = SVector{m}(n .+ (1:m))
#     _inds = SVector{n+m}(xinds)
#     z = SVector{n+m}([x;u])
#     KnotPoint(z, _x, _u, _inds, dt)
# end

function KnotPoint(x::AbstractVector, m::Int)
    u = zeros(m)
    KnotPoint(x, u, 0.)
end

@inline state(z::KnotPoint) = z.z[z._x]
@inline control(z::KnotPoint) = z.z[z._u]
@inline is_terminal(z::KnotPoint) = z.dt == 0

const Traj = Vector{<:KnotPoint}

@inline function discrete_dynamics(model::AbstractModel, z::KnotPoint)
    discrete_dynamics(model, state(z), control(z), z.dt)
end

function propagate_dynamics(model::AbstractModel, z_::KnotPoint, z::KnotPoint)
    x_next = discrete_dynamics(model, z)
    z_.z = [x_next; control(z_)]
end



@inline function discrete_jacobian(model::AbstractModel, z::KnotPoint)
    discrete_jacobian(model, z.z, z.dt)
end

function discrete_dynamics!(f, model, Z::Traj)
    for k in eachindex(Z)
        f[k] = discrete_dynamics(model, Z[k])
    end
end

function discrete_jacobian!(∇f, model, Z::Traj)
    for k in eachindex(∇f)
        ∇f[k] = discrete_jacobian(model, Z[k])
    end
end

function stage_cost(cost::CostFunction, z::KnotPoint)
    if is_terminal(z)
        stage_cost(cost, state(z))
    else
        stage_cost(cost, state(z), control(z), z.dt)
    end
end

function cost(obj::Objective, Z::Traj)::Float64
    J = 0.0
    for k in eachindex(Z)
        J += stage_cost(obj[k], Z[k])
    end
    return J
end


function cost_expansion(cost::CostFunction, z::KnotPoint)
    Qx, Qu, Qxx, Quu, Qux = cost_expansion(cost, state(z), control(z))
    if is_terminal(z)
        dt_x = 1.0
        dt_u = 0.0
    else
        dt_x = z.dt
        dt_u = z.dt
    end
    return Qx*dt_x, Qu*dt_u, Qxx*dt_x, Quu*dt_u, Qux*dt_u
end

function cost_expansion(E, obj::Objective, Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        E.x[k], E.u[k], E.xx[k], E.uu[k], E.ux[k] = cost_expansion(obj[k], Z[k])
    end
end


struct CostExpansion{T,N,M,L1,L2,L3}
    x::Vector{SVector{N,T}}
    u::Vector{SVector{M,T}}
    xx::Vector{SMatrix{N,N,T,L1}}
    uu::Vector{SMatrix{M,M,T,L2}}
    ux::Vector{SMatrix{M,N,T,L3}}
end

function CostExpansion(n,m,N)
    CostExpansion(
        [@SVector zeros(n) for k = 1:N],
        [@SVector zeros(m) for k = 1:N],
        [@SMatrix zeros(n,n) for k = 1:N],
        [@SMatrix zeros(m,m) for k = 1:N],
        [@SMatrix zeros(m,n) for k = 1:N] )
end

function Base.getindex(Q::CostExpansion, k::Int)
    return (x=Q.x[k], u=Q.u[k], xx=Q.xx[k], uu=Q.uu[k], ux=Q.ux[k])
end


struct StaticProblem{L<:AbstractModel,T<:AbstractFloat,N,M,NM}
    model::L
    obj::Objective
    xf::SVector{N,T}
    x0::SVector{N,T}
    Z::Vector{KnotPoint{T,N,M,NM}}
    Z̄::Vector{KnotPoint{T,N,M,NM}}
    N::Int
    dt::T
    tf::T
end

Base.size(prob::StaticProblem{L,T,N,M,NM}) where {L,T,N,M,NM} = (N, M, prob.N)

function rollout!(prob::StaticProblem)
    prob.Z[1].z = [prob.x0; control(prob.Z[1])]
    for k = 2:prob.N
        # discrete_dynamics(prob.model, prob.Z[k], prob.Z[k-1])
        propagate_dynamics(prob.model, prob.Z[k], prob.Z[k-1])
    end
end
