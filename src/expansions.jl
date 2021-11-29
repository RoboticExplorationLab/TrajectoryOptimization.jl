abstract type AbstractExpansion{T} end

struct GradientExpansion{T,N,M} <: AbstractExpansion{T}
    x::SizedVector{N,T,Vector{T}}
    u::SizedVector{M,T,Vector{T}}
    function GradientExpansion{T}(n::Int,m::Int) where T
        new{T,n,m}(SizedVector{n}(zeros(T,n)), SizedVector{m}(zeros(T,m)))
    end
end

# TODO: Move to ALTRO
"""
    DynamicsExpansion{T,N,N̄,M}

Stores the dynamics expansion for a single time instance. 
For a `LieGroupModel`, it will provide access to both the state and state
error Jacobians.

# Constructors
```julia
DynamicsExpansion{T}(n0, n, m)
DynamicsExpansion{T}(n, m)
```
where `n0` is the size of the full state, and `n` is the size of the error state.

# Methods
To evaluate the dynamics Jacobians, use

    dynamics_expansion!(::Type{Q}, D::DynamicsExpansion, model, Z)

To compute the Jacobians for the error state, use
    
    error_expansion!(D::DynamicsExpansion, model, G)

where `G` is a vector of error-state Jacobians. These can be computed using
`RobotDynamics.state_diff_jacobian(G, model, Z)`.

# Extracting Jacobians
The Jacobians should be extracted using

    fdx, fdu = error_expansion(D::DynamicsExpansion, model)

This method will provide the error state Jacobians for `LieGroupModel`s, and 
    the normal Jacobian otherwise. Both `fdx` and `fdu` are a `SizedMatrix`.
"""
struct DynamicsExpansion{T,N,N̄,M}
    f::Vector{T}
    ∇f::Matrix{T} # n × (n+m)
    ∇²f::Matrix{T}  # (n+m) × (n+m)
    A_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    B_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    A::SizedMatrix{N̄,N̄,T,2,Matrix{T}}  # corrected error Jacobian
    B::SizedMatrix{N̄,M,T,2,Matrix{T}}  # corrected error Jacobian
    fxx::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    fuu::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    fux::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    tmpA::SizedMatrix{N,N,T,2,Matrix{T}}
    tmpB::SizedMatrix{N,M,T,2,Matrix{T}}
    tmp::SizedMatrix{N,N̄,T,2,Matrix{T}}
    function DynamicsExpansion{T}(n0::Int, n::Int, m::Int) where T
        f = zeros(n0)
        ∇f = zeros(n0,n0+m)
        ∇²f = zeros(n0+m,n0+m)
        ix = 1:n0
        iu = n0 .+ (1:m)
        A_ = view(∇f, ix, ix)
        B_ = view(∇f, ix, iu)
        A = SizedMatrix{n,n}(zeros(n,n))
        B = SizedMatrix{n,m}(zeros(n,m))
        fxx = view(∇²f, ix, ix)
        fuu = view(∇²f, iu, iu)
        fux = view(∇²f, iu, ix)
        tmpA = SizedMatrix{n0,n0}(zeros(n0,n0))
        tmpB = SizedMatrix{n0,m}(zeros(n0,m))
        tmp = SizedMatrix{n0,n}(zeros(n0,n))
        new{T,n0,n,m}(f,∇f,∇²f,A_,B_,A,B,fxx,fuu,fux,tmpA,tmpB,tmp)
    end
    function DynamicsExpansion{T}(n::Int, m::Int) where T
        f = zeros(n)
        ∇f = zeros(n,n+m)
        ∇²f = zeros(n+m,n+m)
        ix = 1:n
        iu = n .+ (1:m)
        A_ = view(∇f, ix, ix)
        B_ = view(∇f, ix, iu)
        A = SizedMatrix{n,n}(zeros(n,n))
        B = SizedMatrix{n,m}(zeros(n,m))
        fxx = view(∇²f, ix, ix)
        fuu = view(∇²f, iu, iu)
        fux = view(∇²f, iu, ix)
        tmpA = A
        tmpB = B
        tmp = zeros(n,n)
        new{T,n,n,m}(f,∇f,∇²f,A_,B_,A,B,fxx,fuu,fux,tmpA,tmpB,tmp)
    end
end

@inline DynamicsExpansion(model::DiscreteDynamics) = DynamicsExpansion{Float64}(model)
@inline function DynamicsExpansion{T}(model::DiscreteDynamics) where T
    n,m = size(model)
    n̄ = RD.errstate_dim(model)
    DynamicsExpansion{T}(n,n̄,m)
end

function save_tmp!(D::DynamicsExpansion)
    D.tmpA .= D.A_
    D.tmpB .= D.B_
end

function dynamics_expansion!(sig::FunctionSignature, diff::DiffMethod, 
                             model::DiscreteDynamics, D::Vector{<:DynamicsExpansion}, Z::Traj)
    for k in eachindex(D)
        # RobotDynamics.discrete_jacobian!(Q, D[k].∇f, model, Z[k], args...)
        RobotDynamics.jacobian!(sig, diff, model, D[k].∇f, D[k].f, Z[k])
        # save_tmp!(D[k])
        # D[k].tmpA .= D[k].A_  # avoids allocations later
        # D[k].tmpB .= D[k].B_
    end
end

error_expansion!(D::Vector{<:DynamicsExpansion{<:Any,Nx,Ne}}, model::DiscreteDynamics, G) where {Nx,Ne} = 
    error_expansion!(RD.statevectortype(model), D, model, G)

function error_expansion!(::RD.EuclideanState, D::Vector{<:DynamicsExpansion{<:Any,Nx,Ne}}, model::DiscreteDynamics, G) where {Nx,Ne}
    @assert Nx == Ne
    for d in D
        copyto!(d.A, d.A_) 
        copyto!(d.B, d.B_) 
    end
end

function error_expansion!(::RD.RotationState, D::Vector{<:DynamicsExpansion{<:Any,Nx,Ne}}, model::DiscreteDynamics, G) where {Nx,Ne}
    @assert Nx > Ne
    for k in eachindex(D)
        # save_tmp!(D[k])
        error_expansion!(D[k], G[k], G[k+1])
    end
end

# Helper methods
function error_expansion!(D::DynamicsExpansion,G1,G2)
    D.tmpA .= D.A_
    D.tmpB .= D.B_
    mul!(D.tmp, D.tmpA, G1)
    mul!(D.A, Transpose(G2), D.tmp)
    mul!(D.B, Transpose(G2), D.tmpB)
end

# @inline error_expansion(D::DynamicsExpansion, model::DiscreteLieDynamics) = D.A, D.B
@inline error_expansion(D::DynamicsExpansion, model::DiscreteDynamics) = D.A, D.B



struct StaticExpansion{T,N,M,NN,MM,NM}
    x::SVector{N,T}
    xx::SMatrix{N,N,T,NN}
    u::SVector{M,T}
    uu::SMatrix{M,M,T,MM}
    ux::SMatrix{M,N,T,NM}
end

function StaticExpansion(E::AbstractExpansion)
    StaticExpansion(SVector(E.x), SMatrix(E.xx),
        SVector(E.u), SMatrix(E.uu), SMatrix(E.ux))
end

function StaticExpansion(x,xx,u,uu,ux)
    StaticExpansion(SVector(x), SMatrix(xx), SVector(u), SMatrix(uu), SMatrix(ux))
end

"""
    Expansion{n,m,T}

Stores a full second-order expansion of a scalar-valued function (e.g. the cost function)
with respect to both the state and control. The Hessian and gradient with respect to the 
concatenated state and control, available via the `hess` and `grad` fields.

The pieces with respect to the state and control separately are available via 
`x`, `xx`, `u`, `uu`, and `ux`, which are aliased (for backward compatibility) with
`q`, `Q`, `r`, `R`, and `H`.
"""
struct Expansion{n,m,T} <: AbstractMatrix{T}
    # not sure why calling it with 
    data::Matrix{T}
    hess::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}
    grad::SubArray{T, 1, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
    xx::SizedMatrix{n,n,T,2,SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}}
    uu::SizedMatrix{m,m,T,2,SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}}
    ux::SizedMatrix{m,n,T,2,SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}}
    x::SizedVector{n,T,SubArray{T,1,Matrix{T},Tuple{UnitRange{Int64}, Int64}, true}}
    u::SizedVector{m,T,SubArray{T,1,Matrix{T},Tuple{UnitRange{Int64}, Int64}, true}}
    function Expansion{T}(n::Int, m::Int) where T
        ix,iu = 1:n, n .+ (1:m)
        data = zeros(T,n+m,n+m+1)
        hess = view(data,:,1:n+m)
        xx = SizedMatrix{n,n}(view(hess,ix,ix))
        uu = SizedMatrix{m,m}(view(hess,iu,iu))
        ux = SizedMatrix{m,n}(view(hess,iu,ix))
        grad = view(data,:,n+m+1)
        x = SizedVector{n}(view(grad,ix))
        u = SizedVector{m}(view(grad,iu))
        hess .= I(n+m)
        grad .= 0 
        new{n,m,T}(data, hess, grad, xx,uu,ux,x,u)
    end
end
function Base.getproperty(E::Expansion, field::Symbol)
    if field == :q
        getfield(E, :x)
    elseif field == :r
        getfield(E, :u)
    elseif field == :Q
        getfield(E, :xx)
    elseif field == :R
        getfield(E, :uu)
    elseif field == :H
        getfield(E, :ux)
    else
        getfield(E, field)
    end
end

static_expansion(E::Expansion) = StaticExpansion(E.x, E.xx, E.u, E.uu, E.ux)

# Array Interface
Base.size(E::Expansion{n,m}) where {n,m} = (n+m,n+m+1)
@inline Base.getindex(E::Expansion, i::Int) = getindex(E.data, i) 
@inline Base.getindex(E::Expansion, I::Vararg{Int,2}) = getindex(E.data, I...)
@inline Base.setindex!(E::Expansion, v, i::Int) = setindex!(E.data, v, i)
@inline Base.setindex!(E::Expansion, v, I::Vararg{Int,2}) = E.data[I[1],I[2]] = v 
Base.IndexStyle() = IndexLinear()