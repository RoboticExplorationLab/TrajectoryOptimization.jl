
# struct SProblem{L<:AbstractModel,T<:AbstractFloat,NM,N}
#     model::L
#     obj::Objective
#     x0::SVector{N,T}
#     xf::SVector{N,T}
#     Z::Vector{SVector{NM,T}}
#     N::Int
#     dt::T
#     tf::T
#
# end


struct KnotPoint{T,N,M,NM}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T
end

function KnotPoint(x::AbstractVector, u::AbstractVector, dt::Real)
    n = length(x)
    m = length(u)
    _x = SVector{n}(1:n)
    _u = SVector{m}(n .+ (1:m))
    z = SVector{n+m}([x;u])
    KnotPoint(z,_x,_u, dt)
end

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

function cost(obj::Objective, Z::Traj)
    J = 0.0
    for k in eachindex(Z)
        J += stage_cost(obj[k], Z[k])
    end
    return J
end

function cost_expansion(cost::CostFunction, z::KnotPoint)
    Qx, Qu, Qxx, Quu, Qux = cost_expansion(cost, state(z), control(z), z.dt)
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
