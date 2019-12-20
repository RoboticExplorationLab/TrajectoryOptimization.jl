using Makie
using Plots

function AbstractPlotting.lines(X::Vector{<:SVector}, inds=1:length(X[1]))
    A = hcat(X...)
    scene = Scene()
    for i in inds
        lines!(scene, A[i,:], color=:blue)
    end
    scene
end

function Plots.plot(X::Vector{<:SVector}, inds=1:length(X[1]); kwargs...)
    A = hcat(X...)
    Plots.plot(A[inds,:]'; kwargs...)
end
