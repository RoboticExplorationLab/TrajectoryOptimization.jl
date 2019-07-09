export
    Primals

# For Minimum Time DIRCOL
function PartedArrays.create_partition(n::Int,m::Int,q::Int,N::Int,uN=N-1,hN=N-1)
    @assert hN <= uN
    @assert uN <= N
    Nx = N*n
    Nu = uN*m
    Nh = hN*q
    Nz = Nx+Nu+Nh
    ind_x = zeros(Int,n,N)
    ind_u = zeros(Int,m,uN)
    ind_h = zeros(Int,q,hN)
    ix = 1:n
    iu = n .+ (1:m)
    ih = (n+m) .+ (1:q)
    for k = 1:uN
        ind_x[:,k] = ix .+ (k-1)*(n+m+q)
        ind_u[:,k] = iu .+ (k-1)*(n+m+q)
        if k <= hN
            ind_h[:,k] = ih .+ (k-1)*(n+m+q)
        end
    end
    if uN == N-1
        ind_x[:,N] = ix .+ (N-1)*(n+m+q)
    end
    return (X=ind_x, U=ind_u, H=ind_h)
end

struct PrimalsMT{T<:Real}
    Z::Vector{T}
    X::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    U::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    H::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    equal::Bool
end

# function Primals(Z::Vector{T},n::Int,m::Int) where T
#     if length(Z) % (n+m) == 0
#         N = length(Z) ÷ (n+m)
#         uN = N
#     elseif length(Z) % (n+m) == n
#         N = length(Z) ÷ (n+m) + 1
#         uN = N-1
#     end
#     part_z = create_partition(n,m,N,uN)
#     Primals(Z,part_z)
# end

function PrimalsMT(Z,part_z::NamedTuple)
    N, uN, hN = size(part_z.X,2), size(part_z.U,2), size(part_z.H,2)
    X = [view(Z,part_z.X[:,k]) for k = 1:N]
    U = [view(Z,part_z.U[:,k]) for k = 1:uN]
    H = [view(Z,part_z.H[:,k]) for k = 1:hN]

    PrimalsMT(Z,X,U,H, N==uN)
end

"""Create a Primals from vectors of SubArrays.
This is the fastest method to convert a vector Z to subarrays X,U. This will overwrite X and U!
"""
function PrimalsMT(Z::Vector{T},X::Vector{S},U::Vector{S}, H::Vector{S}) where {T,S<:SubArray}
    N = length(X)
    uN = length(U)
    hN = length(H)
    for k = 1:uN
        X[k] = view(Z,X[k].indices[1])
        U[k] = view(Z,U[k].indices[1])
        if k <= hN
            H[k] = view(Z,H[k].indices[1])
        end
    end
    if uN == N-1
        X[N] = view(Z,X[N].indices)
    end
    PrimalsMT(Z,X,U,H, N==uN)
end

function PrimalsMT(Z::Vector{T},Z0::Primals{T}) where T
    X = deepcopy(Z0.X)
    U = deepcopy(Z0.U)
    H = deepcopy(Z0.H)
    uN = length(U)
    hN = length(H)
    N = length(X)
    for k = 1:uN
        X[k] = view(Z,X[k].indices[1])
        U[k] = view(Z,U[k].indices[1])
        if k <= hN
            H[k] = view(Z,H[k].indices[1])
        end
    end
    if uN == N-1
        X[N] = view(Z,X[N].indices)
    end
    PrimalsMT(Z,X,U,H, Z0.equal)
end

""" $(TYPEDSIGNATURES)
Combine state and control trajectories.
This is a little slow and less memory efficient than converting from a the combined vector to individual trajectories.
"""
function PrimalsMT(X::VectorTrajectory{T}, U::VectorTrajectory{T}, H::VectorTrajectory{T}) where T
    N,uN,hN = length(X), length(U), length(H)
    n,m = length(X[1]), length(U[1])
    NN = N*n + uN*m + hN
    Z = zeros(T,NN)
    part_z = create_partition(n,m,1,N,uN,hN)
    Z[part_z.X] = to_array(X)
    Z[part_z.U] = to_array(U)
    Z[part_z.H] = to_array(H)

    PrimalsMT(Z,part_z)
end

function PrimalsMT(prob::Problem{T}, equal::Bool=false) where T
    U = copy(prob.U)
    if equal
        U = push!(U, U[end])
    end
    PrimalsMT(prob.X, U, get_dt_traj(prob))
end


function packMT(prob::Problem{T}) where T
    n,m,N = size(prob)
    part_z = create_partition(n,m,1,N,N,N-1)
    NN = N*(n+m) + (N-1)
    Z = PartedVector(zeros(T,NN),part_z)
    copyto!(Z.X, prob.X)
    copyto!(Z.U, prob.U)
    copyto!(Z.H, get_dt_traj(prob))
    return Z
end

function unpackMT(Z::Vector{<:Real}, part_z::NamedTuple)
    N, uN, hN = size(part_z.X,2), size(part_z.U,2), size(part_z.H,2)
    X = [view(Z,part_z.X[:,k]) for k = 1:N]
    U = [view(Z,part_z.U[:,k]) for k = 1:uN]
    H = [view(Z,part_z.H[:,k]) for k = 1:hN]

    return X, U, H
end

function packMT(X,U,H, part_z)
    n,m = length(X[1]), length(U[1])
    N, uN, hN = length(X), length(U), length(H)
    Z = zeros(eltype(X[1]), N*n + uN*m + hN)
    for k = 1:N
        Z[part_z.X[:,k]] = X[k]
        Z[part_z.U[:,k]] = U[k]
        if k <= hN
            Z[part_z.H[:,k]] = H[k]
        end
    end
    return Z
end

function packMT(X,U,H)
    n = length(X[1])
    m = length(U[1])
    N = length(X)
    uN = length(U)
    hN = length(H)
    part_z = create_partition(n,m,1,N,uN,hN)
    Z = packMT(X,U,H,part_z)
end
