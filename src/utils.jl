import Base: convert, copyto!, Array, vec, println, copy
import LinearAlgebra: norm
import Plots: plot, plot!

function interp_rows(N::Int,tf::Float64,X::AbstractMatrix)::Matrix
    n,N1 = size(X)
    t1 = range(0,stop=tf,length=N1)
    t2 = collect(range(0,stop=tf,length=N))
    X2 = zeros(n,N)
    for i = 1:n
        interp_cubic = CubicSplineInterpolation(t1, X[i,:])
        X2[i,:] = interp_cubic(t2)
    end
    return X2
end

pos(x) = max(0,x)

function to_array(X::Vector{<:AbstractArray})
    N = length(X)
    Y = zeros(size(X[1])...,N)
    ax = axes(Y)
    for i = 1:N
        Y[ax[1:end-1]...,i] = X[i]
    end
    Y
end

function to_array(X::Vector{T}) where T
    return X
end

function vec(A::Trajectory)
	vec(to_array(A))
end

function to_trajectory(X::AbstractArray)
    to_dvecs(X)
end

function to_dvecs(X::AbstractArray)
    N = size(X)[end]
    ax = axes(X)[1:end-1]
    [X[ax...,i] for i = 1:N]
end

"Turn a vector into vectors of vectors of varying length"
function to_dvecs(X::AbstractVector, part::Vector{Int})
	@assert sum(part) == length(X)
	p = insert!(cumsum(part), 1, 0)
	[view(X, p[k]+1:p[k+1]) for k = 1:length(part)]
end

function copyto!(A::Vector{T}, B::Vector{T},n::Int) where T
	N = length(A)
	idx = 1:n
	for k = 1:N
		A[k][idx] = B[k][idx]
	end
end

"Copy trajectory → trajectory"
function copyto!(A::Vector{D}, B::Vector{D}) where {D<:VecOrMat{T} where T}
    A .= copy.(B)
end

"Copy trajectory → array"
function copyto!(A::AbstractMatrix{T}, B::VectorTrajectory{T}) where T
	for k = 1:length(B)
		A[:,k] = B[k]
	end
end

"Copy array → trajectory "
function copyto!(A::VectorTrajectory{T}, B::AbstractMatrix{T}) where T
	for k = 1:length(A)
		A[k] = B[:,k]
	end
end

function root_dir()
    joinpath(dirname(pathof(TrajectoryOptimization)),"..")
end

function ispossemidef(A)
	eigs = eigvals(A)
	if any(real(eigs) .< 0)
		return false
	else
		return true
	end
end

"""
$(SIGNATURES)
Plots a circle given a center and radius
"""
function plot_circle!(center,radius;kwargs...)
    pts = Plots.partialcircle(0,2π,100,radius)
    x,y = Plots.unzip(pts)
    x .+= center[1]
    y .+= center[2]
    plot!(Shape(x,y);kwargs...)
end

function plot_obstacles(circles,clr=:red)
    for circle in circles
        x,y,r = circle
        plot_circle!((x,y),r,color=clr,linecolor=clr,label="")
    end
end

function plot_trajectory!(X::AbstractMatrix;kwargs...)
    plot!(X[1,:],X[2,:];kwargs...)
end
plot_trajectory!(X::AbstractVectorTrajectory; kwargs...) = plot_trajectory!(to_array(X); kwargs...)


function plot_vertical_lines!(p::Plots.Plot,x::Vector; kwargs...)
	ylim = collect(ylims(p))
	plot_vertical_lines!(x, ylim; kwargs...)
end

function plot_vertical_lines!(x,ylim=[-100,100]; kwargs...)
    ys = [ylim for val in x]
    xs = [[val; val] for val in x]
    plot!(xs,ys, linestyle=:dash, color=:black, label=""; kwargs...)
end

plot(X::Trajectory; kwargs...) = plot(to_array(X)'; kwargs...)
plot!(X::Trajectory; kwargs...) = plot!(to_array(X)'; kwargs...)
plot(X::Trajectory, inds::UnitRange; kwargs...) = plot(to_array(X)[inds,:]'; kwargs...)


## Simple constraint primitives
"""
$(SIGNATURES)
Circle constraint function (c ⩽ 0, negative is satisfying constraint)
"""
function circle_constraint(x,x0,y0,r)
	return -((x[1]-x0)^2 + (x[2]-y0)^2  - r^2)
end

circle_constraint(x,c,r) = circle_constraint(x,c[1],c[2],r)

"""
$(SIGNATURES)
Sphere constraint function (c ⩽ 0, negative is satisfying constraint)
"""
function sphere_constraint(x,x0,y0,z0,r)
	return -((x[1]-x0)^2 + (x[2]-y0)^2 + (x[3]-z0)^2  - r^2)
end

function sphere_constraint(x,x0,r)
	return -((x[1]-x0[1])^2 + (x[2]-x0[2])^2 + (x[3]-x0[3])^2-r^2)
end
