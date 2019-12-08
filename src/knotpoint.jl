export
    KnotPoint,
    Traj,
    state,
    control

"""
Stores the states and controls for a single knot point
"""
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

# Constructor for terminal time step
function KnotPoint(x::AbstractVector, m::Int)
    u = zeros(m)
    KnotPoint(x, u, 0.)
end

@inline state(z::KnotPoint) = z.z[z._x]
@inline control(z::KnotPoint) = z.z[z._u]
@inline is_terminal(z::KnotPoint) = z.dt == 0

const Traj = AbstractVector{<:KnotPoint}
traj_size(Z::Vector{<:KnotPoint{T,N,M}}) where {T,N,M} = N,M,length(Z)

function Base.copy(Z::Vector{KnotPoint{T,N,M,NM}}) where {T,N,M,NM}
    [KnotPoint((@SVector ones(NM)) .* z.z, z._x, z._u, z._inds, z.dt) for z in Z]
    # Z_new = [KnotPoint(deepcopy(z.z), z._x, z._u, z._inds, z.dt) for z in Z]
end

function Traj(n::Int, m::Int, dt::AbstractFloat, N::Int, equal=false)
    x = Inf*@SVector ones(n)
    u = @SVector zeros(m)
    Traj(x,u,dt,N,equal)
end

function Traj(x::SVector, u::SVector, dt::AbstractFloat, N::Int, equal=false)
    equal ? uN = N : uN = N-1
    Z = [KnotPoint(x,u,dt) for k = 1:uN]
    if !equal
        m = length(u)
        push!(Z, KnotPoint(x,m))
    end
    return Z
end

function Traj(X::Vector, U::Vector, dt::Vector)
    Z = [KnotPoint(X[k], U[k], dt[k]) for k = 1:length(U)]
    if length(U) == length(X)-1
        push!(Z, KnotPoint(X[end],length(U[1])))
    end
    return Z
end

@inline states(Z::Traj) = state.(Z)
@inline controls(Z::Traj) = control.(Z)

function set_states!(Z::Traj, X)
    for k in eachindex(Z)
        Z[k].z = [X[k]; control(Z[k])]
    end
end

function set_controls!(Z::Traj, U)
    for k in 1:length(Z)-1
        Z[k].z = [state(Z[k]); U[k]]
    end
end
