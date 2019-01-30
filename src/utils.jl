import Base: convert, copyto!, Array
import LinearAlgebra: norm
import Plots: plot, plot!

function get_cost_matrices(solver::Solver)
    solver.obj.Q, solver.obj.R, solver.obj.Qf, solver.obj.xf
end

function get_sizes(solver::Solver)
    return solver.model.n, solver.model.m, solver.N
end

function get_sizes(X::Vector{T}, U::Vector{T}) where {T<:MVector}
    N = length(X)
    n,m = length(X[1]), length(U[1])
end

function get_N(solver::Solver,method::Symbol)
    get_N(solver.N,method)
end

function get_N(N0::Int,method::Symbol)
    if method == :midpoint
        N,N_ = N0,N0
    elseif method == :trapezoid
        N,N_ = N0,N0
    elseif method == :hermite_simpson_separated
        N,N_ = 2N0-1,2N0-1
    elseif method == :hermite_simpson
        N,N_ = N0,2N0-1
    end
end

function get_sizes(X::AbstractArray,U::AbstractArray)
    n,N = size(X)
    m = size(U,1)
    return n,m,N
end

function norm(X::Vector{MVector{S,Float64}}) where {S}
    sqrt(norm2(X))
end

function norm2(X::Union{Vector{MVector{S,Float64} where S}, Vector{Vector{Float64}}})
    v = 0.
    for i = 1:size(X,1)
        v += X[i]'X[i]
    end
    v
end

function norm2(X::Union{MVector{S,Float64},Vector{Float64}}) where {S}
    norm(X)^2
end

function norm(X::Vector{MVector{S,Float64}},inds::UnitRange{Int64}) where {S}
    sqrt(norm2(X,inds))
end

function norm2(X::Union{Vector{MVector{S,Float64}} where S, Vector{Vector{Float64}}},inds::UnitRange{Int64})
    v = 0.
    for i = 1:size(X,1)
        v += X[i][inds]'X[i][inds]
    end
    v
end

function norm2(x::Vector{Real})
    v = 0.
    for val in x
        v += val^2
    end
    return v
end

function to_array(X::Vector{Vector{Float64}})
    N = length(X)
    n = length(X[1])
    Y = zeros(n,N)
    for i = 1:N
        Y[:,i] = X[i]
    end
    Y
end

function to_array(X::Vector{Matrix{Float64}})
    N = length(X)
    n,m = size(X[1])
    Y = zeros(n,m,N)
    for i = 1:N
        Y[:,:,i] = X[i]
    end
    Y
end


function to_array(X::Vector{T}) where {T<:MArray}
    N = length(X)
    Y = zeros(size(X[1])...,N)
    ax = axes(Y)
    for i = 1:N
        Y[ax[1:end-1]...,i] = X[i]
    end
    Y
end

function to_array(A::Vector{D} where {D<:Diagonal})
    n = size(A[1],1)
    N = length(A)
    B = zeros(n,n,N)
    for i = 1:N
        B[:,:,i] = A[i]
    end
    return B
end

function to_dvecs(X::AbstractArray)
    N = size(X)[end]
    ax = axes(X)[1:end-1]
    [X[ax...,i] for i = 1:N]
end

function to_svecs(X::AbstractArray)
    N = size(X)[end]
    s = size(X)[1:end-1]
    ax = axes(X)[1:end-1]
    [MArray{Tuple{s...}}(X[ax...,1]) for i = 1:N]
end

function copyto!(A::Vector{T}, B::AbstractArray{Float64}) where {T<:Union{MArray,VecOrMat}}
    N = size(B)[end]
    ax = axes(B)[1:end-1]
    for i = 1:N
        A[i] = B[ax...,i]
    end
    A
end

function copyto!(A::Vector{T}, B::Vector{T}) where {T<:VecOrMat{Float64}}
    A .= copy.(B)
end

function copyto!(A::Vector{T}, B::Vector{T}) where {T<:SArray{S,Float64,N,L} where {S,N,L}}
    A .= copy.(B)
end

function Array{Float64,3}(X::Vector{D}) where {D<:Diagonal}
    to_array(X)
end

function convert(::Type{Array{Float64,3}}, X::Vector{Diagonal{Float64,V}}) where {V}
    to_array(X)
end

function convert(::Type{Array}, X::Vector{T}) where {T<:MArray}
    to_array(X)
end

function println(level::Symbol, msg::String)::Nothing
    if level_priorities[level] ≥ level_priorities[debug_level::Symbol]
        println(msg)
    end
end

function root_dir()
    joinpath(dirname(pathof(TrajectoryOptimization)),"..")
end

print_info(msg) = println(:info,msg)
print_debug(msg) = println(:debug,msg)

"""
$(SIGNATURES)

    Convert ZYX Euler angles to quaternion (q =[v;s])
"""
function eul2quat(eul)
    ## Translate the given Euler angle with a specified axis rotation sequence into the corresponding quaternion:

    quat = zeros(4,1)
    # case 'ZYX'
	c_1 = cos(eul[1,1]*0.5)
	s_1 = sin(eul[1,1]*0.5)
	c_2 = cos(eul[2,1]*0.5)
	s_2 = sin(eul[2,1]*0.5)
	c_3 = cos(eul[3,1]*0.5)
	s_3 = sin(eul[3,1]*0.5)

    # https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf, the original algorithm has q = [s;v]
	quat[4,1] = c_1*c_2*c_3 + s_1*s_2*s_3
	quat[1,1] = c_1*c_2*s_3 - s_1*s_2*c_3
	quat[2,1] = c_1*s_2*c_3 + s_1*c_2*s_3
	quat[3,1] = s_1*c_2*c_3 - c_1*s_2*s_3

    quat
end

function print_solver(solver::Solver,name::String,io::IO=STDOUT)
    println(io,"###  $name  ###")

    println(io,"\nModel Props")
    println(io,"-----------")
    println(io,"\t n: $(solver.model.n)")
    println(io,"\t m: $(solver.model.m)")
    println(io,"\t inplace dynamics?: $(is_inplace_dynamics(solver.model))")

    println(io,"\nObjective")
    println(io,"----------")
    println(io,"\t tf: $(obj.tf)")
    println(io,"\t x0: $(obj.x0)")
    println(io,"\t xf: $(obj.xf)")
    println(io,"\t Q: $(diag(obj.Q))")
    println(io,"\t R: $(diag(obj.R))")

    println(io,"\nSolver Settings")
    println(io,"-----------------")
    println(io,"\t dt: $(solver.dt)")
    println(io,"\t N: $(solver.N)")
    println(io,"\t integration: $(solver.integration)")

end

"""
$(SIGNATURES)
Generate random circle constraints
"""
function generate_random_circle_obstacle_field(n_circles::Int64,x_rand::Float64=10.0,y_rand::Float64=10.0,r_rand::Float64=0.5)
    x0 = x_rand*rand(n_circles)
    y0 = y_rand*rand(n_circles)
    r = r_rand*ones(n_circles)

    function constraints!(c,x,u)
        for i = 1:n_circles
            c[i] = circle_constraint(x,x0[i],y0[i],r[i])
        end
    end
    return constraints!, zip(x0, y0, r)
end

"""
$(SIGNATURES)
Generate random sphere constraints
"""
function generate_random_sphere_obstacle_field(n_spheres::Int64,x_rand::Float64=10.0,y_rand::Float64=10.0,z_rand::Float64=10.0,r_rand::Float64=0.5)
    x0 = x_rand*rand(n_sphere)
    y0 = y_rand*rand(n_spheres)
    z0 = z_rand*rand(n_spheres)
    r = r_rand*ones(n_spheres)

    function constraints(x,u)::Array
        c = zeros(typeof(x[1]),n_spheres)

        for i = 1:n_spheres
            c[i] = sphere_constraint(x,x0[i],y0[i],z0[i],r[i])
        end
        c
    end
    constraints, (x0, y0, z0, r)
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

function plot_trajectory!(res::TrajectoryOptimization.SolverVectorResults; kwargs...)
	plot_trajectory!(to_array(res.X); kwargs...)
end

"""
$(SIGNATURES)
    Skew-symmetric cross-product matrix (a.k.a. hat map)
"""
function hat(x)
      S = [  0   -x[3] x[2];
	  	    x[3]   0  -x[1];
		   -x[2]  x[1]  0]
end

"""
$(SIGNATURES)
    Convert quaternion (q = [s;v]) to rotation matrix
"""
function quat2rot(q)
      q = q./norm(q)
      s = q[1]
	  v = q[2:4]

      R = Matrix{Float64}(I,3,3) + 2*hat(v)*(hat(v) + s.*Matrix{Float64}(I,3,3))
end


function ispossemidef(A)
	eigs = eigvals(A)
	if any(real(eigs) .< 0)
		return false
	else
		return true
	end
end


function convergence_rate(stats::Dict;tail::Float64=0.5,plot_fit=false)
    total_iters = stats["iterations"]
    start = Int(ceil((1-tail)*total_iters))
    iters = collect(start:total_iters)
    grad = stats["gradient_norm"][iters]
    coeffs = convergence_rate(iters,grad)
    if plot_fit
        x = log.(1:total_iters)
        p = plot(x,log.(stats["gradient_norm"]),xlabel="iterations (log)",ylabel="gradient (log)")
        line = @. coeffs[2]*x + coeffs[1]
        plot!(x,line)
        ylim = collect(ylims(p))
        plot_vertical_lines!(log.(stats["outer_updates"]),ylim,linecolor=:black,linestyle=:dash,label="")
        display(p)
    end
    return coeffs[2]
end

function convergence_rate(x::Vector,y::Vector)
    n = length(x)
    X = [ones(n) log.(x)]
    coeffs = X\log.(y)
    return coeffs
end

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
