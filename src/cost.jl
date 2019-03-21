import Base.copy

#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end

function cost(cost::CostFunction,X::Trajectory,U::Trajectory,dt::AbstractFloat)
    N = length(X)
    J = 0.0
    for k = 1:N-1
        J += stage_cost(cost,X[k],U[k])*dt
    end
    J += stage_cost(cost,X[N])
    return J
end

taylor_expansion(cost::CostFunction,x,u) = taylor_expansion(cost,x,u,1)
stage_cost(cost::CostFunction,x,u) = stage_cost(cost,x,u,1)

"""
$(TYPEDEF)
Cost function of the form
    xₙᵀ Qf xₙ + qfᵀxₙ + ∫ ( xᵀQx + uᵀRu + xᵀHu + q⁠ᵀx  rᵀu ) dt from 0 to tf
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
    (xₙ-x_f)ᵀ Qf (xₙ - x_f) ∫ ( (x-x_f)ᵀQ(x-xf) + uᵀRu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite
"""
function LQRCost(Q,R,Qf,xf)
    H = zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = zeros(size(R,1))
    c = 0.5*xf'*Q*xf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf
    return QuadraticCost(Q, R, H, q, r, c, Qf, qf, cf)
end

function taylor_expansion(cost::QuadraticCost, x::AbstractVector{Float64}, u::AbstractVector{Float64}, k::Int)
    m = get_sizes(cost)[2]
    return cost.Q, cost.R, cost.H, cost.Q*x + cost.q, cost.R*u[1:m] + cost.r
end

function taylor_expansion(cost::QuadraticCost, xN::AbstractVector{Float64})
    return cost.Qf, cost.Qf*xN + cost.qf
end

gradient(cost::QuadraticCost, x::AbstractVector{Float64}, u::AbstractVector{Float64}) = cost.Q*x + cost.q, cost.R*u + cost.r
gradient(cost::QuadraticCost, xN::AbstractVector{Float64}) = cost.Qf*xN + cost.qf

function stage_cost(cost::QuadraticCost, x::AbstractVector, u::AbstractVector, k::Int)
    0.5*x'cost.Q*x + 0.5*u'*cost.R*u + cost.q'x + cost.r'u + cost.c
end

function stage_cost(cost::QuadraticCost, xN::AbstractVector)
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
    function expansion(x,u)
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

function auto_expansion_function(ℓ,ℓf,n,m)
    z = zeros(n+m)
    hess = zeros(n+m,n+m)
    grad = zeros(n+m)
    qf,Qf = zeros(n), zeros(n,n)

    linds = LinearIndices(hess)
    xinds = 1:n
    uinds = n .+(1:m)
    inds = (x=xinds, u=uinds, xx=linds[xinds,xinds], uu=linds[uinds,uinds], ux=linds[uinds,xinds])
    function ℓ_aug(z)
        x = view(z,xinds)
        u = view(z,uinds)
        ℓ(x,u)
    end
    function expansion(x::Vector,u::Vector)
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
    function expansion(xN::Vector)
        ForwardDiff.gradient!(qf,ℓf,xN)
        ForwardDiff.hessian!(Qf,ℓf,xN)
        return Qf, qf
    end
end

function taylor_expansion(cost::GenericCost, x::AbstractVector{Float64}, u::AbstractVector{Float64}, k::Int)
    cost.expansion(x,u)
end

function taylor_expansion(cost::GenericCost, xN::AbstractVector{Float64})
    cost.expansion(xN)
end

# TODO: Split gradient and hessian calculations

stage_cost(cost::GenericCost, x::AbstractVector{Float64}, u::AbstractVector{Float64}, k::Int) = cost.ℓ(x,u)
stage_cost(cost::GenericCost, xN::AbstractVector{Float64}) = cost.ℓf(xN)

get_sizes(cost::GenericCost) = cost.n, cost.m
copy(cost::GenericCost) = GenericCost(copy(cost.ℓ,cost.ℓ,cost.n,cost.m))



"""
$(TYPEDEF)
Cost function of the form
    ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf
"""
struct AugmentedLagrangianCost{T} <: CostFunction
    cost::C where C<:CostFunction
    constraints::ConstraintSet
    λ::PartedVecTrajectory{T}  # Lagrange multipliers
    μ::PartedVecTrajectory{T}  # Penalty Term
    a::PartedVecTrajectory{Bool}  # Active set
    c::PartedVecTrajectory{T}  # Constraint values
    ∇c::PartedMatTrajectory{T}    # Constraint jacobians
end

function update_constraints!(c::Trajectory,constraints::ConstraintSet,X::Trajectory,U::Trajectory)
    N = length(X)
    for k = 1:N-1
        evaluate!(c[k],constraints,X[k],U[k])
    end
    evaluate!(c[N],constraints,X[N])
end

function active_set!(a::Trajectory,c::Trajectory,λ::Trajectory,tol=1e-3)
    N = length(c)
    for k = 1:N
        active_set!(a[k],c[k],λ[k])
    end
end

function active_set!(a::AbstractVector{Bool}, c::AbstractVector, λ::AbstractVector, tol=1e-3)
    # inequality_active!(a,c,λ,tol)
    a.equality .= true
    a.inequality .=  @. (c.inequality >= tol) | (λ.inequality > 0)
    return nothing
end

function active_set(c::AbstractVector, λ::AbstractVector, tol=1e-3)
    a = BlockArray(trues(length(c)),c.parts)
    a.equality .= true
    a.inequality .=  @. (c.inequality >= tol) | (λ.inequality > 0)
    return a
end

function update_Iμ!(Iμ::Trajectory,a::Trajectory,μ::Trajectory)
    N = length(a)
    for k = 1:N
        Iμ[k] .= a[k] .* μ[k]
    end
end

function constraint_cost(c::Trajectory,λ::Trajectory,μ::Trajectory,a::Trajectory)::AbstractFloat
    N = length(c)
    J = 0.0
    for k = 1:N
        a = active_set!(c[k],λ[k])
        J += λ[k]'c[k] + 1/2*c[k]'Diagonal(a .* μ[k])*c[k]
    end
    return J
end

function penalty_cost(c::AbstractVector,λ::AbstractVector,μ::AbstractVector)
    a = active_set(c,λ)
    Iμ = Diagonal(μ.A)
    Iμ = Iμ*a.A
    λ'c + 1/2*c'Diagonal(a .* μ)*c
end

function stage_constraint_cost(alcost::AugmentedLagrangianCost,x,u,k::Int)
    c = alcost.c[k]
    λ = alcost.λ[k]
    μ = alcost.μ[k]
    evaluate!(c,alcost.constraints,x,u)
    penalty_cost(c,λ,μ)
end

function stage_constraint_cost(alcost::AugmentedLagrangianCost,x)
    c = alcost.c[end]
    λ = alcost.λ[end]
    μ = alcost.μ[end]
    evaluate!(c,alcost.constraints,x)
    penalty_cost(c,λ,μ)
end

function stage_cost(alcost::AugmentedLagrangianCost, x::AbstractVector, u::AbstractVector, k)
    J0 = stage_cost(alcost.cost,x,u,k)
    J0 + stage_constraint_cost(alcost,x,u,k)
end

function stage_cost(alcost::AugmentedLagrangianCost, x::AbstractVector)
    J0 = stage_cost(alcost.cost,x)
    J0 + stage_constraint_cost(alcost,x)
end

function cost(alcost::AugmentedLagrangianCost,X::Trajectory,U::Trajectory,dt::AbstractFloat)
    N = length(X)
    J = cost(alcost.cost,X,U,dt)
    for k = 1:N-1
        J += stage_constraint_cost(alcost,X[k],U[k],k)
    end
    J += stage_constraint_cost(alcost,X[N])
    return J
end

function taylor_expansion(alcost::AugmentedLagrangianCost,x,u, k::Int)
    Q,R,H,q,r = taylor_expansion(alcost.cost,x,u,k)

    c = alcost.c[k]
    λ = alcost.λ[k]
    μ = alcost.μ[k]
    a = active_set(c,λ)
    Iμ = Diagonal(a .* μ)
    ∇c = alcost.∇c[k]
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

function taylor_expansion(alcost::AugmentedLagrangianCost,x)
    Qf,qf = taylor_expansion(alcost.cost,x)

    c = alcost.c[N]
    λ = alcost.λ[N]
    μ = alcost.μ[N]
    a = active_set(c,λ)
    Iμ = Diagonal(a .* μ)
    cx = alcost.∇c[N]
    jacobian!(cx,alcost.constraints,x)

    # Second Order pieces
    Qf += cx'Iμ*cx

    # First order pieces
    qf += cx'*(Iμ*c + λ)

    return Qf,qf
end
