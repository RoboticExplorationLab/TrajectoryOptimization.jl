export
    KnotPoint,
    Traj,
    state,
    control

abstract type AbstractKnotPoint{T,N,M,NM} end

""" $(TYPEDEF)
Stores critical information corresponding to each knot point in the trajectory optimization
problem, including the state and control values, as well as the time and time step length.

# Getters
Use the following methods to access values from a `KnotPoint`:
```julia
x  = state(z::KnotPoint)    # returns the n-dimensional state as a SVector
u  = control(z::KnotPoint)  # returns the m-dimensional control vector as a SVector
t  = z.t                    # current time
dt = z.dt                   # time step length
```

# Setters
Use the following methods to set values in a `KnotPoint`:
```julia
set_state!(z::KnotPoint, x)
set_control!(z::KnotPoint, u)
z.t = t
z.dt = dt
```

# Constructors
```julia
KnotPoint(x, u, dt, t=0.0)
KnotPoint(x, m, t=0.0)  # for terminal knot point
```

Use `is_terminal(z::KnotPoint)` to determine if a `KnotPoint` is a terminal knot point (e.g.
has no time step length and z.t == tf).
"""
mutable struct KnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M,NM}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T # time step
    t::T  # total time
end

function KnotPoint(x::AbstractVector, u::AbstractVector, dt::Float64, t=0.0)
    n = length(x)
    m = length(u)
    xinds = ones(Bool, n+m)
    xinds[n+1:end] .= 0
    _x = SVector{n}(1:n)
    _u = SVector{m}(n .+ (1:m))
    z = SVector{n+m}([x;u])
    KnotPoint(z, _x, _u, dt, t)
end

# Constructor for terminal time step
function KnotPoint(x::AbstractVector, m::Int, t=0.0)
    u = zeros(m)
    KnotPoint(x, u, 0., t)
end

@inline state(z::AbstractKnotPoint) = z.z[z._x]
@inline control(z::AbstractKnotPoint) = z.z[z._u]
@inline is_terminal(z::AbstractKnotPoint) = z.dt == 0

const Traj = AbstractVector{<:KnotPoint}
traj_size(Z::Vector{<:KnotPoint{T,N,M}}) where {T,N,M} = N,M,length(Z)

function Base.copy(Z::Vector{KnotPoint{T,N,M,NM}}) where {T,N,M,NM}
    [KnotPoint((@SVector ones(NM)) .* z.z, z._x, z._u, z.dt, z.t) for z in Z]
end

function Traj(n::Int, m::Int, dt::AbstractFloat, N::Int, equal=false)
    x = NaN*@SVector ones(n)
    u = @SVector zeros(m)
    Traj(x,u,dt,N,equal)
end

function Traj(x::SVector, u::SVector, dt::AbstractFloat, N::Int, equal=false)
    equal ? uN = N : uN = N-1
    Z = [KnotPoint(x,u,dt,(k-1)*dt) for k = 1:uN]
    if !equal
        m = length(u)
        push!(Z, KnotPoint(x,m,(N-1)*dt))
    end
    return Z
end

function Traj(X::Vector, U::Vector, dt::Vector, t=cumsum(dt) .- dt[1])
    Z = [KnotPoint(X[k], U[k], dt[k], t[k]) for k = 1:length(U)]
    if length(U) == length(X)-1
        push!(Z, KnotPoint(X[end],length(U[1]),t[end]))
    end
    return Z
end

@inline states(Z::Traj) = state.(Z)
@inline controls(Z::Traj) = control.(Z[1:end-1])

states(Z::Traj, i::Int) = [state(z)[i] for z in Z]

set_state!(z::KnotPoint, x) = z.z = [x; control(z)]
set_control!(z::KnotPoint, u) = z.z = [state(z); u]

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

function set_controls!(Z::Traj, u::SVector)
    for k in 1:length(Z)-1
        Z[k].z = [state(Z[k]); u]
    end
end

function set_times!(Z::Traj, ts)
    for k in eachindex(ts)
        Z[k].t = ts[k]
    end
end

function get_times(Z::Traj)
    [z.t for z in Z]
end

function shift_fill!(Z::Traj)
    N = length(Z)
    for k in eachindex(Z)
        Z[k].t += Z[k].dt
        if k < N
            Z[k].z = Z[k+1].z
        else
            Z[k].t += Z[k-1].dt
        end
    end
end

struct StaticKnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M,NM}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T # time step
    t::T  # total time
end
