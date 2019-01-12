################################################################################
#                                                                              #
#                        DIRCOL VARIABLES STRUCTURE                            #
#                                                                              #
################################################################################

struct DircolVars{T}
    Z::Vector{T}
    X::SubArray{T}
    U::SubArray{T}
end

function DircolVars(Z::Vector,n::Int,m::Int,N::Int)
    z = reshape(Z,n+m,N)
    X = view(z,1:n,:)
    U = view(z,n+1:n+m,:)
    DircolVars(Z,X,U)
end

function DircolVars(n::Int,m::Int,N::Int)
    Z = zeros((n+m)N)
    z = reshape(Z,n+m,N)
    X = view(z,1:n,:)
    U = view(z,n+1:n+m,:)
    DircolVars(Z,X,U)
end

function DircolVars(X::Matrix,U::Matrix)
    Z = packZ(X,U)
    n,m,N = get_sizes(X,U)
    DircolVars(Z,n,m,N)
end

function DircolVars(res::SolverIterResults)
    DircolVars(to_array(res.X), [to_array(res.U) res.U[end]])
end

DiffFloat = Union{Float64,ForwardDiff.Dual}
struct DircolResults <: SolverIterResults
    vars::DircolVars
    # vars_::DircolVars
    Z::Vector{Float64}
    X::SubArray{Float64}
    U::SubArray{Float64}
    fVal::Matrix{Float64}
    X_::Union{Matrix{Float64},SubArray{Float64}}
    U_::Union{Matrix{Float64},SubArray{Float64}}
    fVal_::Matrix{Float64}
    A::Array{Float64,3}
    B::Array{Float64,3}
    weights::Vector{Float64}
    # c_colloc::Matrix{Float64}
    # ceq::Matrix{Float64}
    # c::Matrix{Float64}
    # DircolResults(Z,X,U,fVal,X_,U_,fVal_,A,B,weights) =
    #     new(Z,X,U,fVal,X_,U_,fVal_,A,B,weights)
end
#
# abstract type DircolMethod end
#
# type HermiteSimpson <: DircolMethod end

function DircolResults(solver::Solver,method::Symbol)
    DircolResults(get_sizes(solver)...,method)
end

function DircolResults(n::Int,m::Int,N::Int,method::Symbol)
    N,N_ = get_N(N,method)
    Z = zeros(N*(n+m))
    vars = DircolVars(Z,n,m,N)
    X,U = vars.X,vars.U
    fVal = zero(X)
    if method == :hermite_simpson
        X_ = zeros(n,N_)
        U_ = zeros(m,N_)
        fVal_ = zeros(n,N_)
        A = zeros(n,n,N_)
        B = zeros(n,m,N_)
    elseif method == :midpoint
        X_ = zeros(n,N) # midpoints plus terminal
        U_ = U
        fVal_ = zero(X_)
        N_ = size(X_,2)
        A = zeros(n,n,N_)
        B = zeros(n,m,N_)
    else
        X_ = X
        U_ = U
        fVal_ = fVal
        N_ = size(X_,2)
        A = zeros(n,n+m,N_) # These methods conveniently use the gradient of Z
        B = zeros(0,0,N_)
    end

    weights = get_weights(method,N_)
    DircolResults(vars,Z,X,U,fVal,X_,U_,fVal_,A,B,weights)
end
