export
    Primals

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

# TODO: Inherit from AbstractArray
struct Primals{T<:Real}
    Z::Vector{T}
    X::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    U::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    equal::Bool
end

function Primals(Z::Vector{T},n::Int,m::Int) where T
    if length(Z) % (n+m) == 0
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

"""Create a Primals from vectors of SubArrays.
This is the fastest method to convert a vector Z to subarrays X,U. This will overwrite X and U!
"""
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

function Primals(Z::Vector{T},Z0::Primals{T}) where T
    X = deepcopy(Z0.X)
    U = deepcopy(Z0.U)
    uN = length(U)
    N = length(X)
    for k = 1:uN
        X[k] = view(Z,X[k].indices[1])
        U[k] = view(Z,U[k].indices[1])
    end
    if uN == N-1
        X[N] = view(Z,X[N].indices)
    end
    Primals(Z,X,U, Z0.equal)
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

Base.size(Z::Primals) = length(Z.X[1]), length(Z.U[1]), length(Z.X)
Base.length(Z::Primals) = length(Z.Z)
Base.copy(Z::Primals) = Primals(copy(Z.Z),Z)

function pack(prob::Problem{T}) where T
    n,m,N = size(prob)
    part_z = create_partition(n,m,N,N)
    NN = N*(n+m)
    Z = PartedVector(zeros(T,NN),part_z)
    copyto!(Z.X, prob.X)
    copyto!(Z.U, prob.U)
    return Z
end

function unpack(Z::Vector{<:Real}, part_z::NamedTuple)
    N, uN = size(part_z.X,2), size(part_z.U,2)
    X = [view(Z,part_z.X[:,k]) for k = 1:N]
    U = [view(Z,part_z.U[:,k]) for k = 1:uN]
    return X, U
end

function pack(X,U, part_z)
    n,m = length(X[1]), length(U[1])
    N, uN = length(X), length(U)
    Z = zeros(eltype(X[1]), N*n + uN*m)
    for k = 1:N
        Z[part_z.X[:,k]] = X[k]
        Z[part_z.U[:,k]] = U[k]
    end
    return Z
end

function pack(X,U)
    n = length(X[1])
    m = length(U[1])
    N = length(X)
    uN = length(U)
    part_z = create_partition(n,m,N,uN)
    Z = pack(X,U,part_z)
end


struct PrimalDual{T} <: AbstractArray{T,1}
    V::Vector{T}
    Z::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    X::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    U::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    Y::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    ν::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    λ::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    active_set::Vector{Bool}
    a::Vector{SubArray{Bool,1,Vector{Bool},Tuple{UnitRange{Int}},true}}
end



function PrimalDual(prob::Problem{T}) where T
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    p_colloc = N*n
    p = num_constraints(prob)
    pcum = insert!(cumsum(p),1,0)
    P = sum(p) + p_colloc

    V = zeros(T, NN+P)
    Z = view(V,1:NN)
    Y = view(V,NN .+ (1:P))

    part_z = create_partition(n,m,N)
    X = [view(V,part_z.X[:,k]) for k = 1:N]
    U = [view(V,part_z.U[:,k]) for k = 1:N-1]
    ν = [view(V, NN + (k-1)*n .+ (1:n)) for k = 1:N]
    λ = [view(V, NN + p_colloc + pcum[k] .+ (1:p[k])) for k = 1:N]
    active_set = ones(Bool,P)
    a = [view(active_set, p_colloc + pcum[k] .+ (1:p[k])) for k = 1:N]
    copyto!.(X, prob.X)
    copyto!.(U, prob.U)
    PrimalDual(V, Z, X, U, Y, ν, λ, active_set, a)
end

function PrimalDual(V::Vector, n::Int, m::Int, N::Int, P::Int) where T
    n,m,N = size(prob)
    NN = N*n + (N-1)*m
    p_colloc = N*n
    p = num_constraints(prob)
    pcum = insert!(cumsum(p),1,0)
    P = sum(p) + p_colloc

    Z = view(V,1:NN)
    Y = view(V,NN .+ (1:P))

    part_z = create_partition(n,m,N)
    X = [view(V,part_z.X[:,k]) for k = 1:N]
    U = [view(V,part_z.U[:,k]) for k = 1:N-1]
    ν = [view(V, NN + (k-1)*n .+ (1:n)) for k = 1:N]
    λ = [view(V, NN + p_colloc + pcum[k] .+ (1:p[k])) for k = 1:N]
    active_set = ones(Bool,P)
    a = [view(active_set, p_colloc + pcum[k] .+ (1:p[k])) for k = 1:N]
    copyto!.(X, prob.X)
    copyto!.(U, prob.U)
    PrimalDual(V, Z, X, U, Y, ν, λ, active_set, a)
end


function Base.copy(V::PrimalDual)
    V2 = copy(V.V)
    Z = view(V2, V.Z.indices[1])
    N = length(V.X)
    X = [view(Z, V.X[k].indices[1]) for k = 1:N]
    U = [view(Z, V.U[k].indices[1]) for k = 1:N-1]
    Y = view(V2, V.Y.indices[1])
    ν = [view(V2, V.ν[k].indices[1]) for k = 1:N]
    λ = [view(V2, V.λ[k].indices[1]) for k = 1:N]
    active_set = copy(V.active_set)
    a = [view(active_set, V.a[k].indices[1]) for k = 1:N]
    PrimalDual(V2, Z, X, U, Y, ν, λ, active_set, a)
end


import Base: size, length, getindex, setindex
size(V::PrimalDual) = size(V.V)
length(V::PrimalDual) = length(V.V)
getindex(V::PrimalDual, i::Int) = V.V[i]
setindex!(V::PrimalDual, v, i::Int) = setindex!(V.V, v, i)
IndexStyle(::PrimalDual) = IndexLinear()

+(V::PrimalDual, A::Vector) = begin V.V .+= A; V end

primals(V::PrimalDual) = V.Z
duals(V::PrimalDual) = V.Y
