import Base.copy

#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end

"Calculate (unconstrained) cost for X and U trajectories"
function cost(cost::CostFunction,X::VectorTrajectory{T},U::VectorTrajectory{T},dt::T)::T where T <: AbstractFloat
    N = length(X)
    J = 0.0
    for k = 1:N-1
        J += stage_cost(cost,X[k],U[k])*dt
    end
    J += stage_cost(cost,X[N])
    return J
end

cost_expansion(cost::CostFunction,x::Vector{T},u::Vector{T}) where T = cost_expansion(cost,x,u,1)
stage_cost(cost::CostFunction,x::Vector{T},u::Vector{T}) where T = stage_cost(cost,x,u,1)

"""
$(TYPEDEF)
Cost function of the form
    1/2xₙᵀ Qf xₙ + qfᵀxₙ +  ∫ ( 1/2xᵀQx + 1/2uᵀRu + xᵀHu + q⁠ᵀx  rᵀu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
mutable struct QuadraticCost{TM,TH,TV,T} <: CostFunction
    Q::TM                 # Quadratic stage cost for states (n,n)
    R::TM                 # Quadratic stage cost for controls (m,m)
    H::TH                 # Quadratic Cross-coupling for state and controls (n,m)
    q::TV                 # Linear term on states (n,)
    r::TV                 # Linear term on controls (m,)
    c::T                  # constant term
    Qf::TM                # Quadratic final cost for terminal state (n,n)
    qf::TV                # Linear term on terminal state (n,)
    cf::T                 # constant term (terminal)
    function QuadraticCost(Q::TM, R::TM, H::TH, q::TV, r::TV, c::T, Qf::TM, qf::TV, cf::T) where {TM, TH, TV, T}
        if !isposdef(R)
            err = ArgumentError("R must be positive definite")
            throw(err)
        end
        if !ispossemidef(Q)
            err = ArgumentError("Q must be positive semi-definite")
            throw(err)
        end
        if !ispossemidef(Qf)
            err = ArgumentError("Qf must be positive semi-definite")
            throw(err)
        end
        new{TM,TH,TV,T}(Q,R,H,q,r,c,Qf,qf,cf)
    end
end


"""
$(SIGNATURES)
Cost function of the form
    1/2(xₙ-x_f)ᵀ Qf (xₙ - x_f) + 1/2 ∫ ( (x-x_f)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
function LQRCost(Q::AbstractArray{T},R::AbstractArray{T},Qf::AbstractArray{T},xf::AbstractVector{T}) where T
    H = zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = zeros(size(R,1))
    c = 0.5*xf'*Q*xf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf
    return QuadraticCost(Q, R, H, q, r, c, Qf, qf, cf)
end

"Second-order Taylor expansion of cost function at time step k"
function cost_expansion(cost::QuadraticCost, x::Vector{T}, u::Vector{T}, k::Int) where T
    m = get_sizes(cost)[2]
    return cost.Q, cost.R, cost.H, cost.Q*x + cost.q, cost.R*u[1:m] + cost.r
end

function cost_expansion(cost::QuadraticCost, xN::Vector{T}) where T
    return cost.Qf, cost.Qf*xN + cost.qf
end

"Gradient of the cost function at a single time step"
gradient(cost::QuadraticCost, x::Vector{T}, u::Vector{T}) where T = cost.Q*x + cost.q, cost.R*u + cost.r
gradient(cost::QuadraticCost, xN::Vector{T}) where T = cost.Qf*xN + cost.qf

function stage_cost(cost::QuadraticCost, x::Vector{T}, u::Vector{T}, k::Int) where T
    0.5*x'cost.Q*x + 0.5*u'*cost.R*u + cost.q'x + cost.r'u + cost.c
end

function stage_cost(cost::QuadraticCost, xN::Vector{T}) where T
    0.5*xN'cost.Qf*xN + cost.qf'*xN + cost.cf
end

function get_sizes(cost::QuadraticCost)
    return size(cost.Q,1), size(cost.R,1)
end

function copy(cost::QuadraticCost)
    return QuadraticCost(copy(cost.Q), copy(cost.R), copy(cost.H), copy(cost.q), copy(cost.r), copy(cost.c), copy(cost.Qf), copy(cost.qf), copy(cost.cf))
end

"""
$(TYPEDEF)
Cost function of the form
    ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf
"""
struct GenericCost <: CostFunction
    ℓ::Function             # Stage cost
    ℓf::Function            # Terminal cost
    expansion::Function     # 2nd order Taylor Series Expansion of the form,  Q,R,H,q,r = expansion(x,u)
    n::Int                  #                                                     Qf,qf = expansion(xN)
    m::Int

end

"""
$(SIGNATURES)
Create a Generic Cost, specifying the gradient and hessian of the cost function analytically

# Arguments
* hess: multiple-dispatch function of the form,
    Q,R,H = hess(x,u) with sizes (n,n), (m,m), (m,n)
    Qf = hess(xN) with size (n,n)
* grad: multiple-dispatch function of the form,
    q,r = grad(x,u) with sizes (n,), (m,)
    qf = grad(x,u) with size (n,)

"""
function GenericCost(ℓ::Function, ℓf::Function, grad::Function, hess::Function, n::Int, m::Int)
    function expansion(x::Vector{T},u::Vector{T}) where T
        Q,R,H = hess(x,u)
        q,r = grad(x,u)
        return Q,R,H,q,r
    end
    expansion(xN) = hess(xN), grad(xN)
    GenericCost(ℓ,ℓf, expansion, n,m)
end


"""
$(SIGNATURES)
Create a Generic Cost. Gradient and Hessian information will be determined using ForwardDiff

# Arguments
* ℓ: stage cost function of the form J = ℓ(x,u)
* ℓf: terminal cost function of the form J = ℓ(xN)
"""
function GenericCost(ℓ::Function, ℓf::Function, n::Int, m::Int)
    linds = LinearIndices(zeros(n+m,n+m))
    xinds = 1:n
    uinds = n .+(1:m)
    inds = (x=xinds, u=uinds, xx=linds[xinds,xinds], uu=linds[uinds,uinds], ux=linds[uinds,xinds])
    expansion = auto_expansion_function(ℓ,ℓf,n,m)

    GenericCost(ℓ,ℓf, expansion, n,m)
end

function auto_expansion_function(ℓ::Function,ℓf::Function,n::Int,m::Int)
    z = zeros(n+m)
    hess = zeros(n+m,n+m)
    grad = zeros(n+m)
    qf,Qf = zeros(n), zeros(n,n)

    linds = LinearIndices(hess)
    xinds = 1:n
    uinds = n .+(1:m)
    inds = (x=xinds, u=uinds, xx=linds[xinds,xinds], uu=linds[uinds,uinds], ux=linds[uinds,xinds])
    function ℓ_aug(z::Vector{T}) where T
        x = view(z,xinds)
        u = view(z,uinds)
        ℓ(x,u)
    end
    function expansion(x::Vector{T},u::Vector{T}) where T
        z[inds.x] = x
        z[inds.u] = u
        ForwardDiff.hessian!(hess, ℓ_aug, z)
        Q = view(hess,inds.xx)
        R = view(hess,inds.uu)
        H = view(hess,inds.ux)

        ForwardDiff.gradient!(grad, ℓ_aug, z)
        q = view(grad,inds.x)
        r = view(grad,inds.u)
        return Q,R,H,q,r
    end
    function expansion(xN::Vector{T}) where T
        ForwardDiff.gradient!(qf,ℓf,xN)
        ForwardDiff.hessian!(Qf,ℓf,xN)
        return Qf, qf
    end
end

function cost_expansion(cost::GenericCost, x::Vector{T}, u::Vector{T}, k::Int) where T
    cost.expansion(x,u)
end

function cost_expansion(cost::GenericCost, xN::Vector{T}) where T
    cost.expansion(xN)
end

# TODO: Split gradient and hessian calculations

stage_cost(cost::GenericCost, x::Vector{T}, u::Vector{T}, k::Int) where T = cost.ℓ(x,u)
stage_cost(cost::GenericCost, xN::Vector{T}) where T = cost.ℓf(xN)

get_sizes(cost::GenericCost) = cost.n, cost.m
copy(cost::GenericCost) = GenericCost(copy(cost.ℓ,cost.ℓ,cost.n,cost.m))

"""
$(TYPEDEF)
Cost function of the form
    `` J(X,U) + λ^T c(X,U) + \frac{1}{2} c(X,U)^T I_{\mu} c(X,U)``
    where ``X`` and ``U`` are state and control trajectories, ``J(X,U)`` is the original cost function,
    ``c(X,U)`` is the vector-value constraint function, μ is the penalty parameter, and ``I_{\mu}``
    is a diagonal matrix that whose entries are μ for active constraints and 0 otherwise.

Internally stores trajectories for the Lagrange multipliers.
$(FIELDS)
"""
struct AugmentedLagrangianCost{T} <: CostFunction
    cost::C where C<:CostFunction
    constraints::ConstraintSet
    C::PartedVecTrajectory{T}  # Constraint values
    ∇C::PartedMatTrajectory{T} # Constraint jacobians
    λ::PartedVecTrajectory{T}  # Lagrange multipliers
    μ::PartedVecTrajectory{T}  # Penalty Term
    active_set::PartedVecTrajectory{Bool}  # Active set
end

"""$(TYPEDSIGNATURES)
Create an AugmentedLagrangianCost from another cost function and a set of constraints
    for a problem with N knot points. Allocates new memory for the internal arrays.
"""
function AugmentedLagrangianCost{T}(cost::CostFunction,constraints::ConstraintSet,N::Int;
        μ_init::T=1.,λ_init::T=0.)
    # Get sizes
    n,m = get_sizes(cost)
    C,∇C,λ,μ,active_set = init_constraint_trajectories(constraints,n,m,N)
    AugmentedLagrangianCost(cost,constraint,C,∇C,λ,μ,active_set)
end

"""$(TYPEDSIGNATURES)
Create an AugmentedLagrangianCost from another cost function and a set of constraints
    for a problem with N knot points, specifying the Lagrange multipliers.
    Allocates new memory for the internal arrays.
"""
function AugmentedLagrangianCost{T}(cost::CostFunction{T},constraints::ConstraintSet,
        λ::PartedVecTrajectory{T}; μ_init::T=1.) where T
    # Get sizes
    n,m = get_sizes(cost)
    N = length(λ)
    C,∇C,_,μ,active_set = init_constraint_trajectories(constraints,n,m,N)
    AugmentedLagrangianCost(cost,constraint,C,∇C,λ,μ,active_set)
end

"Update constraints trajectories"
function update_constraints!(c::PartedVecTrajectory{T},constraints::ConstraintSet,X::VectorTrajectory{T},U::VectorTrajectory{T}) where T
    N = length(X)
    for k = 1:N-1
        evaluate!(c[k],constraints,X[k],U[k])
    end
    evaluate!(c[N],constraints,X[N])
end

"Evaluate active set constraints for entire trajectory"
function update_active_set!(a::PartedVecTrajectory{Bool},c::PartedVecTrajectory{T},λ::PartedVecTrajectory{T},tol::T=0.0) where T
    N = length(c)
    for k = 1:N
        active_set!(a[k],c[k],λ[k])
    end
end

"Evaluate active set constraints for a single time step"
function active_set!(a::AbstractVector{Bool}, c::AbstractVector{T}, λ::AbstractVector{T}, tol::T=0.0) where T
    # inequality_active!(a,c,λ,tol)
    a.equality .= true
    a.inequality .=  @. (c.inequality >= tol) | (λ.inequality > 0)
    return nothing
end

function active_set(c::AbstractVector{T}, λ::AbstractVector{T}, tol::T=0.0) where T
    a = BlockArray(trues(length(c)),c.parts)
    a.equality .= true
    a.inequality .=  @. (c.inequality >= tol) | (λ.inequality > 0)
    return a
end

"Cost function terms for Lagrangian and quadratic penalty"
function aula_cost(a::AbstractVector{Bool},c::AbstractVector{T},λ::AbstractVector{T},μ::AbstractVector{T}) where T
    λ'c + 1/2*c'Diagonal(a .* μ)*c
end

function stage_constraint_cost(alcost::AugmentedLagrangianCost{T},x::AbstractVector{T},u::AbstractVector{T},k::Int) where T
    c = alcost.C[k]
    λ = alcost.λ[k]
    μ = alcost.μ[k]
    a = alcost.active_set[k]
    aula_cost(a,c,λ,μ)
end

function stage_constraint_cost(alcost::AugmentedLagrangianCost{T},x::AbstractVector{T}) where T
    c = alcost.C[end]
    λ = alcost.λ[end]
    μ = alcost.μ[end]
    a = alcost.active_set[end]
    aula_cost(a,c,λ,μ)
end

function stage_cost(alcost::AugmentedLagrangianCost{T}, x::AbstractVector{T}, u::AbstractVector{T}, k::Int) where T
    J0 = stage_cost(alcost.cost,x,u,k)
    J0 + stage_constraint_cost(alcost,x,u,k)
end

function stage_cost(alcost::AugmentedLagrangianCost{T}, x::AbstractVector{T}) where T
    J0 = stage_cost(alcost.cost,x)
    J0 + stage_constraint_cost(alcost,x)
end

"Augmented Lagrangian cost for X and U trajectories"
function cost(alcost::AugmentedLagrangianCost{T},X::VectorTrajectory{T},U::VectorTrajectory{T},dt::T) where T <: AbstractFloat
    N = length(X)
    J = cost(alcost.cost,X,U,dt)
    update_constraints!(alcost.C,alcost.constraints,X,U)
    update_active_set!(alcost.active_set,alcost.C,alcost.λ)

    for k = 1:N-1
        J += stage_constraint_cost(alcost,X[k],U[k],k)
    end
    J += stage_constraint_cost(alcost,X[N])
    return J
end

"Second-order expansion of augmented Lagrangian cost"
function cost_expansion(alcost::AugmentedLagrangianCost{T},
        x::AbstractVector{T},u::AbstractVector{T}, k::Int) where T
    Q,R,H,q,r = cost_expansion(alcost.cost,x,u,k)

    c = alcost.C[k]
    λ = alcost.λ[k]
    μ = alcost.μ[k]
    a = active_set(c,λ)
    Iμ = Diagonal(a .* μ)
    ∇c = alcost.∇C[k]
    jacobian!(∇c,alcost.constraints,x,u)
    cx = ∇c.x
    cu = ∇c.u

    # Second Order pieces
    Q += cx'Iμ*cx
    R += cu'Iμ*cu
    H += cu'Iμ*cx

    # First order pieces
    g = (Iμ*c + λ)
    q += cx'g
    r += cu'g

    return Q,R,H,q,r
end

function cost_expansion(alcost::AugmentedLagrangianCost{T},x::AbstractVector{T}) where T
    Qf,qf = cost_expansion(alcost.cost,x)
    N = length(alcost.μ)

    c = alcost.C[N]
    λ = alcost.λ[N]
    μ = alcost.μ[N]
    a = active_set(c,λ)
    Iμ = Diagonal(a .* μ)
    cx = alcost.∇C[N]

    jacobian!(cx,alcost.constraints,x)

    # Second Order pieces
    Qf += cx'Iμ*cx

    # First order pieces
    qf += cx'*(Iμ*c + λ)

    return Qf,qf
end
