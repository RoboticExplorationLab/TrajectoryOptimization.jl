using Interpolations
using StaticArrays
using LinearAlgebra
using Statistics
using Makie

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
    r = map(x) do i
        dx = Interpolations.gradient(itpX, i)[1]
        dy = Interpolations.gradient(itpY, i)[1]
        sqrt(dx^2 + dy^2)
    end
    s = cumsum(r .* w)

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
function localToGlobal(X,Y,S,Φ,s,e)
    itpX = interpolate((S,),X, Gridded(Linear()))
    itpY = interpolate((S,),Y, Gridded(Linear()))
    itpΦ = interpolate((S,),Φ, Gridded(Linear()))
    nomX = itpX.(s)
    nomY = itpY.(s)
    nomΦ = itpΦ.(s)
    x = @. nomX + e*sin(nomΦ)
    y = @. nomY - e*cos(nomΦ)
    return x,y
end
