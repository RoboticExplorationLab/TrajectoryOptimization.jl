############################################################################################
#                              OBJECTIVES                                                  #
############################################################################################

abstract type AbstractObjective end
Base.length(obj::AbstractObjective) = length(obj.cost)
state_dim(obj::AbstractObjective) = throw(ErrorException("state_dim not implemented"))
control_dim(obj::AbstractObjective) = throw(ErrorException("control_dim not implemented"))
get_J(obj::AbstractObjective) = throw(ErrorException("get_J not implemented"))



"""$(TYPEDEF)
Objective: stores stage cost(s) and terminal cost functions

Constructors:
```julia
Objective(cost, N)
Objective(cost, cost_term, N)
Objective(costs::Vector{<:CostFunction}, cost_term)
Objective(costs::Vector{<:CostFunction})
```
"""
struct Objective{C} <: AbstractObjective
    cost::Vector{C}
    J::Vector{Float64}
    const_grad::BitVector
    const_hess::BitVector
    function Objective(cost::Vector{C}) where C <: CostFunction
        N = length(cost)
        J = zeros(N)
        grad = zeros(Bool,N)
        hess = zeros(Bool,N)
        new{C}(cost, J, grad, hess)
    end
end

state_dim(obj::Objective) = state_dim(obj.cost[1])
control_dim(obj::Objective) = control_dim(obj.cost[1])
Base.size(obj::Objective) = (state_dim(obj), control_dim(obj))
@inline ExpansionCache(obj::Objective) = ExpansionCache(obj[1])

"""
    is_quadratic(obj::Objective)

Only valid for a cost expansion, i.e. an objective containing the 2nd order expansion of 
    another objective. Determines if the original objective is a quadratic function, or 
    in other words, if the hessian of the objective is constant. 

For example, if the original cost function is an augmented Lagrangian cost function, the
    result will return true only if all constraints are linear.
"""
is_quadratic(obj::Objective) = all(obj.const_hess)

# Constructors
function Objective(cost::CostFunction,N::Int)
    Objective([cost for k = 1:N])
end

function Objective(cost::CostFunction, cost_terminal::CostFunction, N::Int)
    stage, term = promote(cost, cost_terminal)
    Objective([k < N ? stage : term for k = 1:N])
end

function Objective(cost::Vector{<:CostFunction},cost_terminal::CostFunction)
    N = length(cost) + 1
    Objective([cost...,cost_terminal])
end

# Methods
"Get the vector of costs at each knot point. `sum(get_J(obj))` is equal to the cost"
get_J(obj::Objective) = obj.J

Base.copy(obj::Objective) = Objective(copy.(obj.cost))

Base.getindex(obj::Objective,i::Int) = obj.cost[i]

@inline Base.firstindex(obj::Objective) = firstindex(obj.cost)
@inline Base.lastindex(obj::Objective) = lastindex(obj.cost)
@inline Base.iterate(obj::Objective, start=1) = Base.iterate(obj.cost, start)
@inline Base.eltype(obj::Objective) = eltype(obj.cost)
@inline Base.length(obj::Objective) = length(obj.cost)
Base.IteratorSize(obj::Objective) = Base.HasLength()
Base.eachindex(obj::Objective) = Base.OneTo(length(obj))

Base.show(io::IO, obj::Objective{C}) where C = print(io,"Objective")

"""
    CostExpansion{n,m,T}

A vector of `Expansion`s, combined with some bit vectors for storing whether the 
terms are constant or not.

# Constructors
    CostExpansion{T}(n,m,N)
    CostExpansion(n,m,T)  # defaults to Float64
    CostExpansion(E::CostExpansion, model)

where the last constructor is for allocating the expansion for the "raw" expansion, 
    before accounting for the Lie group structure. For a standard (Euclidiean) state 
    vector, it will create an alias to the original expansion. For a `LieGroupModel`,
    it will allocate new storage.
"""
struct CostExpansion{n,m,T} <: AbstractArray{Expansion{n,m,T},1}
    data::Vector{Expansion{n,m,T}}
    const_hess::BitVector
    const_grad::BitVector
    function CostExpansion{T}(n::Int, m::Int, N::Int) where T
        data = [Expansion{T}(n,m) for k = 1:N]
        const_hess = BitVector(undef, N)
        const_grad = BitVector(undef, N)
        new{n,m,T}(data, const_hess, const_grad)
    end
end
@inline CostExpansion(n,m,N) = CostExpansion{Float64}(n,m,N)
function CostExpansion(E::CostExpansion, model::AbstractModel)
    # Create QuadraticObjective linked to error cost expansion
    @assert RobotDynamics.state_diff_size(model) == size(model)[1]
    return E 
end

function CostExpansion(E::CostExpansion{n,m,T}, model::LieGroupModel) where {n,m,T}
    # Create an expansion for the full state dimension
    @assert length(E[1].q) == RobotDynamics.state_diff_size(model)
    n0 = state_dim(model)
    return CostExpansion{T}(n0,m,length(E))
end
Base.size(E::CostExpansion) = size(E.data) 
Base.getindex(E::CostExpansion, i::Int) = Base.getindex(E.data, i)



# ############################################################################################
# #                            Quadratic Objectives (Expansions)
# ############################################################################################
# const QuadraticObjective{n,m,T} = Objective{QuadraticCost{n,m,T,SizedMatrix{n,n,T,2,Matrix{T}},SizedMatrix{m,m,T,2,Matrix{T}}}}
# const QuadraticExpansion{n,m,T} = Objective{<:QuadraticCostFunction{n,m,T}}
# const DiagonalCostFunction{n,m,T} = Union{DiagonalCost{n,m,T},QuadraticCost{n,m,T,<:Diagonal,<:Diagonal}}

# function QuadraticObjective(n::Int, m::Int, N::Int, isequal::Bool=false)
#     Objective([QuadraticCost{Float64}(n,m, terminal=(k==N) && !isequal) for k = 1:N])
# end

# function QuadraticObjective(obj::QuadraticObjective, model::AbstractModel)
#     # Create QuadraticObjective linked to error cost expansion
#     @assert RobotDynamics.state_diff_size(model) == size(model)[1]
#     return obj
# end

# function QuadraticObjective(obj::QuadraticObjective, model::LieGroupModel)
#     # Create QuadraticObjective partially linked to error cost expansion
#     @assert length(obj[1].q) == RobotDynamics.state_diff_size(model)
#     n,m = size(model)
#     costfuns = map(obj.cost) do costfun
#         Q = SizedMatrix{n,n}(zeros(n,n))
#         R = costfun.R
#         H = SizedMatrix{m,n}(zeros(m,n))
#         q = @MVector zeros(n)
#         r = costfun.r
#         c = costfun.c
#         QuadraticCost(Q,R,H,q,r,c, checks=false, terminal=costfun.terminal)
#     end
#     Objective(costfuns)
# end


# Convenience constructors
@doc raw"""
    LQRObjective(Q, R, Qf, xf, N)

Create an objective of the form
`` (x_N - x_f)^T Q_f (x_N - x_f) + \sum_{k=0}^{N-1} (x_k-x_f)^T Q (x_k-x_f) + u_k^T R u_k``

Where `eltype(obj) <: DiagonalCost` if `Q`, `R`, and `Qf` are
    `Union{Diagonal{<:Any,<:StaticVector}}, <:StaticVector}`
"""
function LQRObjective(Q::AbstractArray, R::AbstractArray, Qf::AbstractArray,
        xf::AbstractVector, N::Int; checks=true, uf=@SVector zeros(size(R,1)))
    @assert size(Q,1) == length(xf)
    @assert size(Qf,1) == length(xf)
    @assert size(R,1) == length(uf)
    n = size(Q,1)
    m = size(R,1)
    H = SizedMatrix{m,n}(zeros(m,n))
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = QuadraticCost(Q, R, H, q, r, c, checks=checks)
    ℓN = QuadraticCost(Qf, R, H, q, r, cf, checks=false, terminal=true)

    Objective(ℓ, ℓN, N)
end

function LQRObjective(
        Q::Union{<:Diagonal, <:AbstractVector},
        R::Union{<:Diagonal, <:AbstractVector},
        Qf::Union{<:Diagonal, <:AbstractVector},
        xf::AbstractVector, N::Int;
        uf=(@SVector zeros(size(R,1))),
        checks=true)
    n,m = size(Q,1), size(R,1)
    @assert size(Q,1) == length(xf)
    @assert size(Qf,1) == length(xf)
    @assert size(R,1) == length(uf)
    Q,R,Qf = Diagonal(Q), Diagonal(R), Diagonal(Qf)
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = DiagonalCost(Q, R, q, r, c, checks=checks, terminal=false)

    ℓN = DiagonalCost(Qf, R, qf, r, cf, checks=false, terminal=true)

    Objective(ℓ, ℓN, N)
end

function TrackingObjective(Q,R,Z::AbstractTrajectory; Qf=Q)
    costs = map(Z) do z
        LQRCost(Q, R, state(z), control(z))
    end
    costs[end] = LQRCost(Qf, R, state(Z[end]))
    Objective(costs)
end

function update_trajectory!(obj::Objective{<:QuadraticCostFunction}, Z::AbstractTrajectory, start=1)
    inds = (start-1) .+ (1:length(obj))
    for (i,k) in enumerate(inds)
        set_LQR_goal!(obj[i], state(Z[k]), control(Z[k]))
    end
end