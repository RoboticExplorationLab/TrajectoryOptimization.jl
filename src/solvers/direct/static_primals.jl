
struct StaticPrimals{T<:Real,N,M}
    Z::Vector{T}
    xinds::Vector{SVector{N,Int}}
    uinds::Vector{SVector{M,Int}}
    equal::Bool
end

function StaticPrimals(n::Int, m::Int, N::Int, equal=false)
    NN = n*N + m*(N-1) + equal*m
    Z = zeros(NN)
    uN = N-1 + equal

    xinds = [SVector{n}((n+m)*(k-1) .+ (1:n)) for k = 1:N]
    uinds = [SVector{m}(n + (n+m)*(k-1) .+ (1:m)) for k = 1:N]
    StaticPrimals(Z,xinds,uinds,equal)
end

function Base.copy(P::StaticPrimals)
    StaticPrimals(copy(P.Z),P.xinds,P.uinds,P.equal)
end

function Base.copyto!(P::StaticPrimals, Z::Traj)
    uN = P.equal ? length(Z) : length(Z)-1
    for k in 1:uN
        inds = [P.xinds[k]; P.uinds[k]]
        P.Z[inds] = Z[k].z
    end
    if !P.equal
        P.Z[P.xinds[end]] = state(Z[end])
    end
    return nothing
end

function Base.copyto!(V::AbstractVector{<:Real}, Z::Traj,
        xinds::Vector{<:AbstractVector}, uinds::Vector{<:AbstractVector})
    n,m,N = traj_size(Z)
    equal = (n+m)*N == length(V)

    uN = equal ? N : N-1
    for k in 1:uN
        inds = [xinds[k]; uinds[k]]
        V[inds] = Z[k].z
    end
    if !equal
        V[xinds[end]] = state(Z[end])
    end
    return nothing
end

function Base.copyto!(Z::Traj, P::StaticPrimals)
    uN = P.equal ? length(Z) : length(Z)-1
    for k in 1:uN
        inds = [P.xinds[k]; P.uinds[k]]
        Z[k].z = P.Z[inds]
    end
    if !P.equal
        xN = P.Z[P.xinds[end]]
        Z[end].z = [xN; control(Z[end])]
    end
    return nothing
end

function Base.copyto!(Z::Traj, V::Vector{<:Real},
        xinds::Vector{<:AbstractVector}, uinds::Vector{<:AbstractVector})
    n,m,N = traj_size(Z)
    equal = (n+m)*N == length(V)

    uN = equal ? N : N-1
    for k in 1:uN
        inds = [xinds[k]; uinds[k]]
        Z[k].z = V[inds]
    end
    if !equal
        xN = V[xinds[end]]
        Z[end].z = [xN; control(Z[end])]
    end
    return nothing
end
