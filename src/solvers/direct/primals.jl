

function PartedArrays.create_partition(n::Int,m::Int,N::Int,uN=N-1)
    Nx = N*n
    Nu = uN*m
    Nz = Nx+Nu
    ind_x = zeros(Int,n,N)
    ind_u = zeros(Int,m,uN)
    ix = 1:n
    iu = n .+ (1:m)
    for k = 1:uN
        ind_x[:,k] = ix .+ (k-1)*(n+m)
        ind_u[:,k] = iu .+ (k-1)*(n+m)
    end
    if uN == N-1
        ind_x[:,N] = ix .+ (N-1)*(n+m)
    end
    return (X=ind_x, U=ind_u)
end

struct Primals{T<:Real}
    Z::Vector{T}
    X::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    U::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    equal::Bool
end

function Primals(Z::Vector{T},n::Int,m::Int) where T
    if length(Z) % (n+m) == 0
        println("got here")
        N = length(Z) ÷ (n+m)
        uN = N
    elseif length(Z) % (n+m) == n
        N = length(Z) ÷ (n+m) + 1
        uN = N-1
    end
    part_z = create_partition(n,m,N,uN)
    Primals(Z,part_z)
end

function Primals(Z,part_z::NamedTuple)
    N, uN = size(part_z.X,2), size(part_z.U,2)
    X = [view(Z,part_z.X[:,k]) for k = 1:N]
    U = [view(Z,part_z.U[:,k]) for k = 1:uN]
    Primals(Z,X,U, N==uN)
end

function Primals(Z::Vector{T},X::Vector{S},U::Vector{S}) where {T,S<:SubArray}
    N = length(X)
    uN = length(U)
    for k = 1:uN
        X[k] = view(Z,X[k].indices[1])
        U[k] = view(Z,U[k].indices[1])
    end
    if uN == N-1
        X[N] = view(Z,X[N].indices)
    end
    Primals(Z,X,U, N==uN)
end

""" $(TYPEDSIGNATURES)
Combine state and control trajectories.
This is a little slow and less memory efficient than converting from a the combined vector to individual trajectories.
"""
function Primals(X::VectorTrajectory{T}, U::VectorTrajectory{T}) where T
    N,uN = length(X), length(U)
    n,m = length(X[1]), length(U[1])
    NN = N*n + uN*m
    Z = zeros(T,NN)
    part_z = create_partition(n,m,N,uN)
    Z[part_z.X] = to_array(X)
    Z[part_z.U] = to_array(U)
    Primals(Z,part_z)
end

function Primals(prob::Problem{T}, equal::Bool=false) where T
    U = copy(prob.U)
    if equal
        U = push!(U, U[end])
    end
    Primals(prob.X, U)
end

function packZ(prob::Problem{T}) where T
    n,m,N = size(prob)
    part_z = create_partition(n,m,N,N)
    NN = N*(n+m)
    Z = PartedVector(zeros(T,NN),part_z)
    copyto!(Z.X, prob.X)
    copyto!(Z.U, prob.U)
    return Z
end
