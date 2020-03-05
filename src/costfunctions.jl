import Base.copy

export
    CostFunction,
    QuadraticQuatCost,
    RBCost,
    QuatLQRCost,
    SatDiffCost,
    ErrorQuadratic



#*********************************#
#       COST FUNCTION CLASS       #
#*********************************#

abstract type CostFunction end


#######################################################
#              COST FUNCTION INTERFACE                #
#######################################################

"""
$(TYPEDEF)
Cost function of the form
    1/2xₙᵀ Qf xₙ + qfᵀxₙ +  ∫ ( 1/2xᵀQx + 1/2uᵀRu + xᵀHu + q⁠ᵀx + rᵀu ) dt from 0 to tf
R must be positive definite, Q and Qf must be positive semidefinite

Constructor use any of the following constructors:
```julia
QuadraticCost(Q, R, H, q, r, c)
QuadraticCost(Q, R; H, q, r, c)
QuadraticCost(Q, q, c)
```
Any optional or omitted values will be set to zero(s).
"""
mutable struct QuadraticCost{TQ,TR,TH,Tq,Tr,T} <: CostFunction
    Q::TQ                 # Quadratic stage cost for states (n,n)
    R::TR                 # Quadratic stage cost for controls (m,m)
    H::TH                 # Quadratic Cross-coupling for state and controls (n,m)
    q::Tq                 # Linear term on states (n,)
    r::Tr                 # Linear term on controls (m,)
    c::T                                 # constant term
    function QuadraticCost(Q::TQ, R::TR, H::TH,
            q::Tq, r::Tr, c::T; checks=true) where {TQ,TR,TH,Tq,Tr,T}
        @assert size(Q,1) == length(q)
        @assert size(R,1) == length(r)
        @assert size(H) == (length(r), length(q))
        if checks
            if !isposdef(Array(R))
                @warn "R is not positive definite"
            end
            if !ispossemidef(Array(Q))
                err = ArgumentError("Q must be positive semi-definite")
                throw(err)
            end
        end
        new{TQ,TR,TH,Tq,Tr,T}(Q,R,H,q,r,c)
    end
end

state_dim(cost::QuadraticCost) = length(cost.q)
control_dim(cost::QuadraticCost) = length(cost.r)

# Constructors
function QuadraticCost(Q,R; H=similar(Q,size(R,1), size(Q,1)), q=zeros(size(Q,1)),
        r=zeros(size(R,1)), c=0.0, checks=true)
    QuadraticCost(Q,R,H,q,r,c, checks=checks)
end

function QuadraticCost(Q::Union{Diagonal{T,SVector{N,T}}, SMatrix{N,N,T}},
            R::Union{Diagonal{T,SVector{M,T}}, SMatrix{M,M,T}};
            H=(@SMatrix zeros(T,M,N)),
            q=(@SVector zeros(T,N)),
            r=(@SVector zeros(M)),
            c=0.0, checks=true) where {T,N,M}
    QuadraticCost(Q,R,H,q,r,c, checks=checks)
end

function QuadraticCost(Q,q,c; checks=true)
    QuadraticCost(Q,zeros(0,0),zeros(0,size(Q,1)),q,zeros(0),c,checks=checks)
end

"""
$(SIGNATURES)
Cost function of the form
``(x-x_f)^T Q (x_x_f) + u^T R u``
R must be positive definite, Q must be positive semidefinite
"""
function LQRCost(Q::AbstractArray, R::AbstractArray,
        xf::AbstractVector, uf=(@SVector zeros(size(R,1))); checks=true)
    H = @SMatrix zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    return QuadraticCost(Q, R, H, q, r, c, checks=checks)
end

"""
$(SIGNATURES)
Cost function of the form
``(x-x_f)^T Q (x-x_f)``
Q must be positive semidefinite
"""
function LQRCostTerminal(Qf::AbstractArray,xf::AbstractVector)
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf
    return QuadraticCost(Qf,zeros(0,0),zeros(0,size(Qf,1)),qf,zeros(0),cf)
end

# Cost function methods
function stage_cost(cost::QuadraticCost, x::AbstractVector, u::AbstractVector)
    0.5*x'cost.Q*x + 0.5*u'*cost.R*u + cost.q'x + cost.r'u + cost.c #+ u'*cost.H*x
end

function stage_cost(cost::QuadraticCost, xN::AbstractVector{T}) where T
    0.5*xN'cost.Q*xN + cost.q'*xN + cost.c
end

# function gradient(cost::QuadraticCost, x, u)
#     Qx = cost.Q*x + cost.q + cost.H'*u
#     Qu = cost.R*u + cost.r + cost.H*x
#     return Qx, Qu
# end
#
# function hessian(cost::QuadraticCost, x, u)
#     Qxx = cost.Q
#     Quu = cost.R
#     Qux = cost.H
#     return Qxx, Quu, Qux
# end

function gradient!(E::AbstractExpansion, cost::QuadraticCost, x, u)
    E.x .= cost.Q*x + cost.q + cost.H'*u
    E.u .= cost.R*u + cost.r + cost.H*x
    return nothing
    mul!(E.x, cost.Q, x)
    E.x .+= cost.q
    mul!(E.x, cost.H', u, 1.0, 1.0)

    mul!(E.u, cost.R, u)
    E.u .+= cost.r
    mul!(E.u, cost.H, x, 1.0, 1.0)
    return nothing
end

function hessian!(E::AbstractExpansion, cost::QuadraticCost, x, u)
    E.xx .= cost.Q
    E.uu .= cost.R
    E.ux .= cost.H
    return nothing
end



# Additional Methods
function Base.show(io::IO, cost::QuadraticCost)
    print(io, "QuadraticCost{...}")
end

import Base: +
"Add two Quadratic costs of the same size"
function +(c1::QuadraticCost, c2::QuadraticCost)
    @assert state_dim(c1) == state_dim(c2)
    @assert control_dim(c1) == control_dim(c2)
    QuadraticCost(c1.Q + c2.Q, c1.R + c2.R, c1.H + c2.H,
                  c1.q + c2.q, c1.r + c2.r, c1.c + c2.c)
end

function copy(cost::QuadraticCost)
    return QuadraticCost(copy(cost.Q), copy(cost.R), copy(cost.H), copy(cost.q), copy(cost.r), copy(cost.c))
end




############################################################################################
#                                 QUADRATIC QUATERNION COST FUNCTION
############################################################################################

struct QuadraticQuatCost{T,N,M,N4} <: CostFunction
    Q::Diagonal{T,SVector{N,T}}
    R::Diagonal{T,SVector{M,T}}
    q::SVector{N,T}
    r::SVector{M,T}
    c::T
    w::T
    q_ref::SVector{4,T}
    q_ind::SVector{4,Int}
    Iq::SMatrix{N,4,T,N4}
    function QuadraticQuatCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
            q::SVector{N,T}, r::SVector{M,T}, c::T, w::T,
            q_ref::SVector{4,T}, q_ind::SVector{4,Int}) where {T,N,M}
        Iq = @MMatrix zeros(N,4)
        for i = 1:4
            Iq[q_ind[i],i] = 1
        end
        Iq = SMatrix{N,4}(Iq)
        return new{T,N,M,N*4}(Q, R, q, r, c, w, q_ref, q_ind, Iq)
    end
end


state_dim(::QuadraticQuatCost{T,N,M}) where {T,N,M} = N
control_dim(::QuadraticQuatCost{T,N,M}) where {T,N,M} = M

function QuadraticQuatCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}};
        q=(@SVector zeros(N)), r=(@SVector zeros(M)), c=zero(T), w=one(T),
        q_ref=(@SVector [1.0,0,0,0]), q_ind=(@SVector [4,5,6,7])) where {T,N,M}
    QuadraticQuatCost(Q, R, q, r, c, q_ref, q_ind)
end

function stage_cost(cost::QuadraticQuatCost, x::SVector, u::SVector)
    stage_cost(cost, x) + 0.5*u'cost.R*u + cost.r'u
end

function stage_cost(cost::QuadraticQuatCost, x::SVector)
    J = 0.5*x'cost.Q*x + cost.q'x + cost.c
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    J += cost.w*min(1+dq, 1-dq)
end

# function cost_expansion(cost::QuadraticQuatCost, z::KnotPoint, G)
#     # Gradient
#     Qx = cost.Q*x + cost.q
#     q = x[cost.q_ind]
#     quat = UnitQuaternion(q)
#     dq = cost.q_ref'q
#     if dq < 0
#         Qx += cost.Iq*cost.q_ref*G
#     else
#         Qx -= cost.Iq*cost.q_ref*G
#     end
#     Qu = cost.R*u + cost.r
#
#     # Hessian
#     Q = cost.Q
#     Qxx = G'Q*G
#
#     return Qx, Qu
# end

function gradient(cost::QuadraticQuatCost{T,N,M}, x::SVector, u::SVector) where {T,N,M}
    Qx = cost.Q*x + cost.q
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    if dq < 0
        Qx += cost.w*cost.Iq*cost.q_ref
    else
        Qx -= cost.w*cost.Iq*cost.q_ref
    end
    Qu = cost.R*u + cost.r
    return Qx, Qu
end

function hessian(cost::QuadraticQuatCost, x::SVector{N}, u::SVector{M}) where {N,M}
    Qxx = cost.Q
    Quu = cost.R
    Qux = @SMatrix zeros(M,N)
    return Qxx, Quu, Qux
end

function QuatLQRCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}}, xf,
        uf=(@SVector zeros(M)); w=one(T), quat_ind=(@SVector [4,5,6,7])) where {T,N,M}
    r = -R*uf
    q = -Q*xf
    c = 0.5*xf'Q*xf + 0.5*uf'R*uf
    q_ref = xf[quat_ind]
    return QuadraticQuatCost(Q, R, q, r, c, w, q_ref, quat_ind)
end

function change_dimension(cost::QuadraticQuatCost, n, m)
    n0,m0 = state_dim(cost), control_dim(cost)
    Q_diag = diag(cost.Q)
    R_diag = diag(cost.R)
    q = cost.q
    r = cost.r
    if n0 != n
        dn = n - n0  # assumes n > n0
        pad = @SVector zeros(dn)
        Q_diag = [Q_diag; pad]
        q = [q; pad]
    end
    if m0 != m
        dm = m - m0  # assumes m > m0
        pad = @SVector zeros(dm)
        R_diag = [R_diag; pad]
        r = [r; pad]
    end
    QuadraticQuatCost(Diagonal(Q_diag), Diagonal(R_diag), q, r, cost.c, cost.w,
        cost.q_ref, cost.q_ind)
end

function (+)(cost1::QuadraticQuatCost, cost2::QuadraticCost)
    @assert state_dim(cost1) == state_dim(cost2)
    @assert control_dim(cost1) == control_dim(cost2)
    @assert norm(cost2.H) ≈ 0
    QuadraticQuatCost(cost1.Q + cost2.Q, cost1.R + cost2.R,
        cost1.q + cost2.q, cost1.r + cost2.r, cost1.c + cost2.c,
        cost1.w, cost1.q_ref, cost1.q_ind)
end

(+)(cost1::QuadraticCost, cost2::QuadraticQuatCost) = cost2 + cost1




struct ErrorQuadratic{Rot,N,M} <: CostFunction
    model::RigidBody{Rot}
    Q::Diagonal{Float64,SVector{12,Float64}}
    R::Diagonal{Float64,SVector{M,Float64}}
    r::SVector{M,Float64}
    c::Float64
    x_ref::SVector{N,Float64}
    q_ind::SVector{4,Int}
end


state_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = N
control_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = M

function ErrorQuadratic(model::RigidBody{Rot}, Q::Diagonal{T,<:SVector{12}},
        R::Diagonal{T,<:SVector{M}},
        x_ref::SVector{N}, u_ref=(@SVector zeros(T,M)); r=(@SVector zeros(T,M)), c=zero(T),
        q_ind=(@SVector [4,5,6,7])) where {T,N,M,Rot}
    r += -R*u_ref
    c += 0.5*u_ref'R*u_ref
    return ErrorQuadratic{Rot,N,M}(model, Q, R, r, c, x_ref, q_ind)
end

function stage_cost(cost::ErrorQuadratic, x::SVector)
    dx = state_diff(cost.model, x, cost.x_ref)
    return 0.5*dx'cost.Q*dx + cost.c
end

function stage_cost(cost::ErrorQuadratic, x::SVector, u::SVector)
    stage_cost(cost, x) + 0.5*u'cost.R*u + cost.r'u
end

function cost_expansion(cost::ErrorQuadratic{Rot}, model::AbstractModel,
        z::KnotPoint{T,N,M}, G) where {T,N,M,Rot<:UnitQuaternion}
    x,u = state(z), control(z)
    model = cost.model
    Q = cost.Q
    q = orientation(model, x)
    q_ref = orientation(model, cost.x_ref)
    dq = SVector(q_ref\q)
    err = state_diff(model, x, cost.x_ref)
    dx = @SVector [err[1],  err[2],  err[3],
                    dq[1],   dq[2],   dq[3],   dq[4],
                   err[7],  err[8],  err[9],
                   err[10], err[11], err[12]]
    G = state_diff_jacobian(model, dx) # n × dn

    # Gradient
    dmap = inverse_map_jacobian(model, dx) # dn × n
    Qx = G'dmap'Q*err
    Qu = cost.R*u

    # Hessian
    ∇jac = inverse_map_∇jacobian(model, dx, Q*err)
    Qxx = G'dmap'Q*dmap*G + G'∇jac*G + ∇²differential(model, x, dmap'Q*err)
    Quu = cost.R
    Qux = @SMatrix zeros(M,N-1)
    return Qxx, Quu, Qux, Qx, Qu
end

function cost_expansion(cost::ErrorQuadratic, model::AbstractModel,
        z::KnotPoint{T,N,M}, G) where {T,N,M}
    x,u = state(z), control(z)
    model = cost.model
    q = orientation(model, x)
    q_ref = orientation(model, cost.x_ref)
    err = state_diff(model, x, cost.x_ref)
    dx = err
    G = state_diff_jacobian(model, dx) # n × n

    # Gradient
    dmap = inverse_map_jacobian(model, dx) # n × n
    Qx = G'dmap'cost.Q*err
    Qu = cost.R*u + cost.r

    # Hessian
    Qxx = G'dmap'cost.Q*dmap*G
    Quu = cost.R
    Qux = @SMatrix zeros(M,N)
    return Qxx, Quu, Qux, Qx, Qu
end

function change_dimension(cost::ErrorQuadratic, n, m)
    n0,m0 = state_dim(cost), control_dim(cost)
    Q_diag = diag(cost.Q)
    R_diag = diag(cost.R)
    r = cost.r
    if n0 != n
        dn = n - n0  # assumes n > n0
        pad = @SVector zeros(dn) # assume the new states don't have quaternions
        Q_diag = [Q_diag; pad]
    end
    if m0 != m
        dm = m - m0  # assumes m > m0
        pad = @SVector zeros(dm)
        R_diag = [R_diag; pad]
        r = [r; pad]
    end
    ErrorQuadratic(cost.model, Diagonal(Q_diag), Diagonal(R_diag), r, cost.c,
        cost.x_ref, cost.q_ind)
end

function (+)(cost1::ErrorQuadratic, cost2::QuadraticCost)
    @assert control_dim(cost1) == control_dim(cost2)
    @assert norm(cost2.H) ≈ 0
    @assert norm(cost2.q) ≈ 0
    if state_dim(cost2) == 13
        rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
        Q2 = Diagonal(diag(cost2.Q)[rm_quat])
    else
        Q2 = cost2.Q
    end
    ErrorQuadratic(cost1.model, cost1.Q + Q2, cost1.R + cost2.R,
        cost1.r + cost2.r, cost1.c + cost2.c,
        cost1.x_ref, cost1.q_ind)
end

(+)(cost1::QuadraticCost, cost2::ErrorQuadratic) = cost2 + cost1




struct SatDiffCost{Rot} <: CostFunction
    model::RigidBody{Rot}
    Q1::Diagonal{Float64,SVector{3,Float64}}
    Q2::Diagonal{Float64,SVector{3,Float64}}
    R::Diagonal{Float64,SVector{3,Float64}}
    q_ref::SVector{4,Float64}
    ω_ref::SVector{3,Float64}
end

function stage_cost(cost::SatDiffCost{Rot}, x::SVector) where Rot
    ω = @SVector [x[1],x[2],x[3]]
    q = Dynamics.orientation(cost.model, x)
    q0 = Rot(UnitQuaternion(cost.q_ref))

    dω = ω - cost.ω_ref
    dq = q ⊖ q0
    return 0.5*(dq'cost.Q2*dq + dω'cost.Q1*dω)
end

function stage_cost(cost::SatDiffCost, x::SVector, u::SVector)
    J = stage_cost(cost, x) + 0.5*u'cost.R*u
end


function cost_expansion(cost::SatDiffCost{Rot}, model::AbstractModel,
        z::KnotPoint{T,N,M}, G) where {T,N,M,Rot<:UnitQuaternion}
    x,u = state(z), control(z)
    Q = cost.Q2  # cost for quaternion
    ω = @SVector [x[1],x[2],x[3]]
    q = @SVector [x[1],x[2],x[3],x[4]]
    q = Dynamics.orientation(cost.model, x)
    q0 = Rot(UnitQuaternion(cost.q_ref))

    dω = ω - cost.ω_ref
    dq = q0\q
    err = q ⊖ q0
    G = ∇differential(dq) #Lmult(dq)*Vmat()'

    # Gradient
    Qω = cost.Q1*ω
    dmap = inverse_map_jacobian(dq) #jacobian(CayleyMap,dq)
    Qq = G'dmap'Q*err
    Qx = [Qω; Qq]
    Qu = cost.R*u

    # Hessian
    Qωω = cost.Q1
    ∇jac = inverse_map_∇jacobian(dq, Q*err)
    Qqq = G'dmap'Q*dmap*G + G'∇jac*G
    Qxx = @SMatrix [
        Qωω[1,1] 0 0 0 0 0;
        0 Qωω[2,2] 0 0 0 0;
        0 0 Qωω[3,3] 0 0 0;
        0 0 0 Qqq[1,1] Qqq[1,2] Qqq[1,3];
        0 0 0 Qqq[2,1] Qqq[2,2] Qqq[2,3];
        0 0 0 Qqq[3,1] Qqq[3,2] Qqq[3,3];
    ]
    Quu = cost.R
    Qux = @SMatrix zeros(M,N-1)
    return Qxx, Quu, Qux, Qx, Qu
end

function cost_expansion(cost::SatDiffCost{Rot}, model::AbstractModel,
        z::KnotPoint{T,N,M}, G) where {T,N,M,Rot}
    x,u = state(z), control(z)
    Q = cost.Q2  # cost for quaternion
    ω = @SVector [x[1],x[2],x[3]]
    q = @SVector [x[1],x[2],x[3],x[4]]
    q = Dynamics.orientation(cost.model, x)
    q0 = Rot(UnitQuaternion(cost.q_ref))

    dω = ω - cost.ω_ref
    dq = q0\q
    err = q ⊖ q0
    G = ∇differential(dq) #Lmult(dq)*Vmat()'

    # Gradient
    Qω = cost.Q1*ω
    Qq = G'Q*err
    Qx = [Qω; Qq]
    Qu = cost.R*u

    # Hessian
    Qωω = cost.Q1
    ∇jac = ∇²differential(dq, Q*err)
    Qqq = G'Q*G + ∇jac
    Qxx = @SMatrix [
        Qωω[1,1] 0 0 0 0 0;
        0 Qωω[2,2] 0 0 0 0;
        0 0 Qωω[3,3] 0 0 0;
        0 0 0 Qqq[1,1] Qqq[1,2] Qqq[1,3];
        0 0 0 Qqq[2,1] Qqq[2,2] Qqq[2,3];
        0 0 0 Qqq[3,1] Qqq[3,2] Qqq[3,3];
    ]
    Quu = cost.R
    Qux = @SMatrix zeros(M,N)
    return Qxx, Quu, Qux, Qx, Qu
end

#
# """
# $(TYPEDEF)
# Cost function of the form
#     ℓf(xₙ) + ∫ ℓ(x,u) dt from 0 to tf
# """
# struct GenericCost <: CostFunction
#     ℓ::Function             # Stage cost
#     ℓf::Function            # Terminal cost
#     expansion::Function     # 2nd order Taylor Series Expansion of the form,  Q,R,H,q,r = expansion(x,u)
#     n::Int                  #                                                     Qf,qf = expansion(xN)
#     m::Int
# end
#
# """
# $(SIGNATURES)
# Create a Generic Cost, specifying the gradient and hessian of the cost function analytically
#
# # Arguments
# * hess: multiple-dispatch function of the form,
#     Q,R,H = hess(x,u) with sizes (n,n), (m,m), (m,n)
#     Qf = hess(xN) with size (n,n)
# * grad: multiple-dispatch function of the form,
#     q,r = grad(x,u) with sizes (n,), (m,)
#     qf = grad(x,u) with size (n,)
#
# """
# function GenericCost(ℓ::Function, ℓf::Function, grad::Function, hess::Function, n::Int, m::Int)
#     @warn "Use GenericCost with caution. It is untested and not likely to work"
#     function expansion(x::Vector{T},u::Vector{T}) where T
#         Q,R,H = hess(x,u)
#         q,r = grad(x,u)
#         return Q,R,H,q,r
#     end
#     expansion(xN) = hess(xN), grad(xN)
#     GenericCost(ℓ,ℓf, expansion, n,m)
# end
#
# """ $(TYPEDEF)
# This is an experimental cost function type that allows a cost function to be evaluated
#     on only a portion of the state and control. Right now, the implementation assumes
#     there is no coupling between the state and control.
#
# It should be noted that for `QuadraticCost`s it is likely more efficient to simply make a new
#     `QuadraticCost` that has zeros in the right places, and then add the cost functions together.
#
# # Constructor
# ```julia
# IndexedCost(cost, ix::UnitRange, iu::UnitRange)
# ```
# """
# struct IndexedCost{iX,iU,C} <: CostFunction
#     cost::C
# end
#
# function IndexedCost(cost::C, ix::UnitRange, iu::UnitRange) where C<:CostFunction
#     if C <: QuadraticCost
#         if norm(cost.H) != 0
#             throw(ErrorException("IndexedCost of functions with x-u coupling not implemented"))
#         end
#     else
#         @warn "IndexedCost will only work for costs without x-u coupling (Qux = 0)"
#     end
#     IndexedCost{ix,iu,C}(cost)
# end
#
# @generated function stage_cost(costfun::IndexedCost{iX,iU}, x::SVector{N}, u::SVector{M}) where {iX,iU,N,M}
#     ix = SVector{length(iX)}(iX)
#     iu = SVector{length(iU)}(iU)
#     quote
#         x0 = x[$ix]
#         u0 = u[$iu]
#         stage_cost(costfun.cost, x0, u0)
#     end
# end
#
# @generated function stage_cost(costfun::IndexedCost{iX,iU}, x::SVector{N}) where {iX,iU,N}
#     ix = SVector{length(iX)}(iX)
#     quote
#         x0 = x[$ix]
#         stage_cost(costfun.cost, x0)
#     end
# end
#
# @generated function gradient(costfun::IndexedCost{iX,iU}, x::SVector{N}, u::SVector{M}) where {iX,iU,N,M}
#     l1x = iX[1] - 1
#     l2x = N-iX[end]
#     l1u = iU[1] - 1
#     l2u = M-iU[end]
#     quote
#         x = x[$iX]
#         u = u[$iU]
#         Qx, Qu = gradient(costfun.cost, x, u)
#         Qx = [@SVector zeros($l1x); Qx; @SVector zeros($l2x)]
#         Qu = [@SVector zeros($l1u); Qu; @SVector zeros($l2u)]
#         return Qx, Qu
#     end
# end
#
#
# @generated function hessian(costfun::IndexedCost{iX,iU}, x::SVector{N}, u::SVector{M}) where {iX,iU,N,M}
#     l1x = iX[1] - 1
#     l2x = N-iX[end]
#     l1u = iU[1] - 1
#     l2u = M-iU[end]
#     quote
#         x = x[$iX]
#         u = u[$iU]
#         Qxx, Quu, Qux  = hessian(costfun.cost, x, u)
#         Qxx1 = Diagonal(@SVector zeros($l1x))
#         Qxx2 = Diagonal(@SVector zeros($l2x))
#         Quu1 = Diagonal(@SVector zeros($l1u))
#         Quu2 = Diagonal(@SVector zeros($l2u))
#
#         Qxx = blockdiag(Qxx1, Qxx, Qxx2)
#         Quu = blockdiag(Quu1, Quu, Quu2)
#         Qux = @SMatrix zeros(M,N)
#         Qxx, Quu, Qux
#     end
# end
#
function SparseArrays.blockdiag(Qs::Vararg{<:Diagonal})
    Diagonal(vcat(diag.(Qs)...))
end

function SparseArrays.blockdiag(Qs::Vararg{<:AbstractMatrix})
    # WARNING: this is slow and is only included as a fallback
    cat(Qs...,dims=(1,2))
end
#
# function change_dimension(cost::CostFunction,n,m)
#     n0,m0 = state_dim(cost), control_dim(cost)
#     ix = 1:n0
#     iu = 1:m0
#     IndexedCost(cost, ix, iu)
# end
#
function change_dimension(cost::QuadraticCost, n, m)
    n0,m0 = state_dim(cost), control_dim(cost)
    @assert n >= n0
    @assert m >= m0

    ix = 1:n0
    iu = 1:m0

    Q_ = Diagonal(@SVector zeros(n-n0))
    R_ = Diagonal(@SVector zeros(m-m0))
    H1 = @SMatrix zeros(m0, n-n0)
    H2 = @SMatrix zeros(m-m0, n)
    q_ = @SVector zeros(n-n0)
    r_ = @SVector zeros(m-m0)
    c = cost.c

    # Insert old values
    Q = blockdiag(cost.Q, Q_)
    R = blockdiag(cost.R, R_)
    H = [cost.H H1]
    H = [H; H2]
    q = [cost.q; q_]
    r = [cost.r; r_]
    QuadraticCost(Q,R,H,q,r,c,checks=false)
end
