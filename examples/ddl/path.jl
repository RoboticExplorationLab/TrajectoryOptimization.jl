using Interpolations
using StaticArrays
using LinearAlgebra
using Statistics
using QuadGK
using Plots
# using Makie

function Rot2(θ)
    sθ,cθ = sincos(θ)
    @SMatrix [cθ -sθ; sθ cθ]
end

abstract type AbstractPath end

struct CarPath{T} <: AbstractPath
    s::Vector{T}
    ϕ::Vector{T}
    κ::Vector{T}
    X::Vector{T}
    Y::Vector{T}
end

function curvature(path::CarPath, s)
    k = searchsortedfirst(path.s, s)
    return path.κ[k]
end

struct CirclePath <: AbstractPath
    r::Float64
end
curvature(path::CirclePath, s) = 1/path.r


""" ArcPath """
struct ArcPath <: AbstractPath
    ang0::Float64
    radius::Float64
    ang::Float64
    xc::Float64
    yc::Float64
    function ArcPath(ang0, radius, ang)
        xc = -radius*sin(ang0)
        yc =  radius*cos(ang0)
        new(ang0, radius, ang, xc, yc)
    end
end

function position_change(arc::ArcPath)
    localToGlobal(arc, total_length(arc), 0)
end
total_length(arc::ArcPath) = arc.radius * arc.ang
curvature(arc::ArcPath, s) = 1/arc.radius
final_angle(arc::ArcPath) = arc.ang0 + arc.ang
function localToGlobal(arc::ArcPath, s, e)
    r = arc.radius
    θ = s ./ r
    radius = r .- e
    # radius = r
    ϕ = θ .- π/2 .+ arc.ang0
    x = radius .* cos.(ϕ) .+ arc.xc
    y = radius .* sin.(ϕ) .+ arc.yc
    return x,y
end

""" Straight Path """
struct StraightPath <: AbstractPath
    len::Float64
    ang::Float64
end

function position_change(path::StraightPath)
    x = path.len * cos(path.ang)
    y = path.len * sin(path.ang)
    return x,y
end
total_length(path::StraightPath) = path.len
final_angle(path::StraightPath) = path.ang
curvature(path::StraightPath, s) = 0.0
function localToGlobal(path::StraightPath, s, e)
    Rot2(path.ang)*@SVector [s, e]
end

"Build a straight path that continues from an arc"
function StraightPath(arc::ArcPath, len)
    StraightPath(len, final_angle(arc))
end

"Build an arc that continues from a line"
function ArcPath(line::StraightPath, radius, ang)
    ArcPath(final_angle(line), radius, ang)
end

""" Dubins Path """
struct DubinsPath{P} <: AbstractPath
    paths::P
    lengths::Vector{Float64}
    ds::Vector{Float64}
    dx::Vector{Float64}
    dy::Vector{Float64}
    curvature::Vector{Float64}
end

function DubinsPath(paths)
    num_paths = length(paths)
    lengths = cumsum([total_length(p) for p in paths])
    ds = circshift(lengths, 1)
    ds[1] = 0
    dx,dy = zeros(num_paths), zeros(num_paths)
    for i = 2:num_paths
        dx[i],dy[i] = position_change(paths[i-1])
    end
    dx = cumsum(dx)
    dy = cumsum(dy)
    κ = [curvature(path,0.0) for path in paths]
    DubinsPath(paths, lengths, ds, dx, dy, κ)
end

total_length(path::DubinsPath) = path.lengths[end]
final_angle(path::DubinsPath) = final_angle(path.paths[end])
function curvature(path::DubinsPath, s)
    i = match_segment(path, s)::Int
    return path.curvature[i]
end

function match_segment(path::DubinsPath, s::AbstractVector)
    [searchsortedfirst(path.ds, s_) for s_ in s]
end

function match_segment(path::DubinsPath, s::Real)
    j = searchsortedlast(path.ds, s)
    min(j, length(path.ds))
end

function localToGlobal(path::DubinsPath, s, e)
    x,y = zero(s), zero(s)
    for i in eachindex(s)
        j = match_segment(path, s[i])
        s_ = s[i] - path.ds[j]
        x[i],y[i] = localToGlobal(path.paths[j], s_, e[i])
        x[i] += path.dx[j]
        y[i] += path.dy[j]
    end
    return x,y
end

function Plots.plot(path::AbstractPath; kwargs...)
    s = range(0, total_length(path), length=101)
    e = zero(s)
    x,y = localToGlobal(path, s, e)
    plot(x,y; kwargs...)
end

function Plots.plot!(path::AbstractPath; kwargs...)
    s = range(0, total_length(path), length=101)
    e = zero(s)
    x,y = localToGlobal(path, s, e)
    plot!(x,y; kwargs...)
end


"""
Get a trajectory of s, ϕ, κ from a set of E,N pairs defining the path, where
    s is the distance along the path
    ϕ is the heading wrt the X-axis (E)
    κ is the curvature of the road, in 1/m
"""
function pathToLocal(X,Y)
    # Parameterize the curve from 0 to 1
    N = length(X)
    t = range(0,1,length=N)

    # Create cubic interpolations
    itpX = CubicSplineInterpolation(t,X)
    itpY = CubicSplineInterpolation(t,Y)

    # Compute the gradient vectors
    g = map(t) do i
        dx = Interpolations.gradient(itpX, i)[1]
        dy = Interpolations.gradient(itpY, i)[1]
        @SVector [dx,dy]
    end

    # Compute the hessian vectors
    h = map(t) do i
        ddx = Interpolations.hessian(itpX, i)[1]
        ddy = Interpolations.hessian(itpY, i)[1]
        @SVector [ddx,ddy]
    end

    # Calculate the heading
    ϕ = map(x->atan(x[2],x[1]), g)

    # Calculate the curvature
    κ = map(1:N) do k
        r′ = g[k]
        r″ = h[k]
        k = norm(r′ × r″) / norm(r′)^3
    end

    # Calulcate the path length: ∫ₐᵇ ‖r′(t)‖ dt
    x,w = gauss(N,0,1)
    @show any(isnan,x)
    r = map(x) do i
        dx = Interpolations.gradient(itpX, i)[1]
        dy = Interpolations.gradient(itpY, i)[1]
        sqrt(dx^2 + dy^2)
    end
    s = cumsum(r .* w)

    # Make the first path length zero
    s[1] = 0.0

    return s, ϕ, κ
end

"""
Attempt to recover the E,N coordanates from a set of local coordinates.
Doesn't work super well due to integration drift
"""
function localToGlobal(s,ϕ,κ)
    N = length(s)
    @assert length(ϕ) == N
    @assert length(κ) == N

    X = zeros(N)
    Y = zeros(N)
    for k = 2:N
        r = 1/mean(κ[k-1:k])
        ds = s[k] - s[k-1]
        θ = ds/r
        psi = mean(ϕ[k-1:k])
        psi = ϕ[k-1]
        xc = X[k-1] - r*sin(psi)
        yc = Y[k-1] + r*cos(psi)
        X[k] = xc + r*sin(psi + θ)
        Y[k] = yc - r*cos(psi + θ)
    end
    return X,Y
end

"""
Get the E,N coordinate given the distance along the path (s) and the longitudinal error (e)
    given a nominal path described by X,Y,S,Φ
Inputs s,e can be vectors or scalars
"""
function localToGlobal(path::CarPath, s, e)
    plot(path.X,path.Y, aspect_ratio=:equal)
    X,Y,S,Φ = path.X, path.Y, path.s, path.ϕ
    itpX = interpolate((S,),X, Gridded(Linear()))
    itpY = interpolate((S,),Y, Gridded(Linear()))
    itpΦ = interpolate((S,),Φ, Gridded(Linear()))
    nomX = itpX.(s)
    nomY = itpY.(s)
    nomΦ = itpΦ.(s)
    x = @. nomX - e*sin(nomΦ)
    y = @. nomY + e*cos(nomΦ)
    return x,y
end


function localToGlobal(path::CirclePath, s, e)
    r = path.r
    t = range(0,2pi,length=101)
    X,Y = r*cos.(t), r*sin.(t)
    plot(X,Y, aspect_ratio=:equal)
    θ = s ./ r
    radius = r .+ e
    x = radius .* cos.(θ)
    y = radius .* sin.(θ)
    return x,y
end
