using StaticArrays
using LinearAlgebra
using Test
using Random
using ForwardDiff
const TO = TrajectoryOptimization

# Unit Quaternions
q = rand(UnitQuaternion)
q0 = rand(UnitQuaternion)
qval = SVector(q)
ϕ = @SVector zeros(3)

function mycost(q::SVector{4})
    return 1 - SVector(q0)'q
end
function mycostdiff(v::SVector{3})
    mycost(SVector(q*VectorPart(v)))
end

grad_q = ForwardDiff.gradient(mycost, qval)
hess_q = ForwardDiff.hessian(mycost, qval)
G = TO.∇differential(q)

@test ForwardDiff.gradient(mycostdiff, ϕ) ≈ G'grad_q
@test ForwardDiff.hessian(mycostdiff, ϕ) ≈ G'hess_q*G + TO.∇²differential(q, grad_q)
dq = SVector(q0)'SVector(q)
@test ForwardDiff.gradient(mycostdiff, ϕ) ≈ -G'SVector(q0)
@test ForwardDiff.gradient(mycostdiff, ϕ) ≈ -G'SVector(q0)
G'SVector(q0) ≈ Vmat()*Lmult(q)'SVector(q0)
@test ForwardDiff.hessian(mycostdiff, ϕ) ≈ I(3)*dq

# MRPs
g = rand(MRP)
gval = SVector(g)
g0 = rand(MRP)
dg = @SVector zeros(3)
function mycost(g::SVector{3})
    g = MRP(g)
    dg = g ⊖ g0
    return norm(dg)
end
mycostdiff(dg::SVector{3}) = mycost(SVector(g*MRP(dg)))
mycost(gval)
mycostdiff(dg)
G = TO.∇differential(g)
dfdp = ForwardDiff.gradient(mycost,gval)
dfdp2 = ForwardDiff.hessian(mycost,gval)

ForwardDiff.gradient(mycostdiff,dg) ≈ G'dfdp
ForwardDiff.hessian(mycostdiff,dg) ≈ G'dfdp2*G + TO.∇²differential(g,dfdp)


# State diff cost (Quaternion)
Random.seed!(1)
q = rand(UnitQuaternion)
q0 = rand(UnitQuaternion)
qval = SVector(q)
ϕ = @SVector zeros(3)

RMAP = CayleyMap
function mycost(q::SVector{4})
    q = UnitQuaternion{RMAP}(q)
    err = RMAP(q0\q)
    0.5*err'err
end
function mycostdiff(ϕ::SVector{3})
    mycost(SVector(q*RMAP(ϕ)))
end

mycostdiff(ϕ) ≈ mycost(qval)
grad_q = ForwardDiff.gradient(mycost, qval)
hess_q = ForwardDiff.hessian(mycost, qval)
dq = q0\q
err = CayleyMap(dq)
G = TO.∇differential(dq)
dmap = jacobian(CayleyMap,dq)
∇jac = TO.∇jacobian(CayleyMap, dq, err)
TO.∇²differential(q, grad_q)

@test grad_q ≈ Lmult(q0)*dmap'err
@test ForwardDiff.gradient(mycostdiff,ϕ) ≈ (dmap*G)'err
@test ForwardDiff.hessian(mycostdiff,ϕ) ≈ G'dmap'dmap*G + G'∇jac*G + TO.∇²differential(q,grad_q)

RMAP = MRPMap
@test mycostdiff(ϕ) ≈ mycost(qval)
grad_q = ForwardDiff.gradient(mycost, qval)
hess_q = ForwardDiff.hessian(mycost, qval)
dq = q0\q
err = RMAP(dq)
G = TO.∇differential(dq)
dmap = jacobian(RMAP,dq)
∇jac = TO.∇jacobian(RMAP, dq, err)

@test grad_q ≈ Lmult(q0)*dmap'err
@test ForwardDiff.gradient(mycostdiff,ϕ) ≈ (dmap*G)'err
@test ForwardDiff.hessian(mycostdiff,ϕ) ≈ G'dmap'dmap*G + G'∇jac*G + TO.∇²differential(q,grad_q)

RMAP = VectorPart
@test mycostdiff(ϕ) ≈ mycost(qval)
grad_q = ForwardDiff.gradient(mycost, qval)
hess_q = ForwardDiff.hessian(mycost, qval)
dq = q0\q
err = RMAP(dq)
G = TO.∇differential(dq)
dmap = jacobian(RMAP,dq)
∇jac = TO.∇jacobian(RMAP, dq, err)

@test grad_q ≈ Lmult(q0)*dmap'err
@test ForwardDiff.gradient(mycostdiff,ϕ) ≈ (dmap*G)'err
@test ForwardDiff.hessian(mycostdiff,ϕ) ≈ G'dmap'dmap*G + G'∇jac*G + TO.∇²differential(q,grad_q)

RMAP = ExponentialMap
@test mycostdiff(ϕ) ≈ mycost(qval)
grad_q = ForwardDiff.gradient(mycost, qval)
hess_q = ForwardDiff.hessian(mycost, qval)
dq = q0\q
err = RMAP(dq)
G = TO.∇differential(dq)
dmap = jacobian(RMAP,dq)
∇jac = TO.∇jacobian(RMAP, dq, err)

@test grad_q ≈ Lmult(q0)*dmap'err
ForwardDiff.gradient(mycostdiff,ϕ)
(dmap*G)'err
@test ForwardDiff.hessian(mycostdiff,ϕ) ≈ G'dmap'dmap*G + G'∇jac*G + TO.∇²differential(q,grad_q)

jacobian(ExponentialMap, q)
b = @SVector rand(3)
q0 = ExponentialMap(1e-7*@SVector ones(3))
TO.vecnorm(q0)
jacobian(ExponentialMap, q)'b ≈ TO.∇jacobian(ExponentialMap, q, b)
jacobian(ExponentialMap, q)'b
ForwardDiff.jacobian(x->jacobian(ExponentialMap, UnitQuaternion(x))'b, SVector(q0)) ≈ TO.∇jacobian(ExponentialMap, q0, b)
ForwardDiff.jacobian(x->jacobian(ExponentialMap, UnitQuaternion(x))'b, SVector(q)) ≈
    TO.∇jacobian(ExponentialMap, q, b)
@btime TO.∇jacobian(ExponentialMap, $q, $b)


# MRPs
p = rand(MRP)
p0 = rand(MRP)
pval = SVector(p)
ϕ = @SVector zeros(3)
Q = Diagonal(@SVector rand(3))

function mycost(p::SVector{3})
    p = MRP(p)
    err = SVector(p0\p)
    0.5*err'Q*err
end
function mycostdiff(ϕ::SVector{3})
    mycost(SVector(p*MRP(ϕ)))
end

grad_p = ForwardDiff.gradient(mycost, pval)
hess_p = ForwardDiff.hessian(mycost, pval)
err = SVector(p0\p)

grad_p ≈ TO.∇err(p0,p)'Q*err
hess_p ≈ TO.∇²err(p0,p,Q*err) + TO.∇err(p0,p)'Q*TO.∇err(p0,p)

grad_d = ForwardDiff.gradient(mycostdiff, ϕ)
hess_d = ForwardDiff.hessian(mycostdiff, ϕ)
@test grad_d ≈ TO.∇differential(p0\p)'Q*err
@test hess_d ≈ TO.∇differential(p0\p)'Q*TO.∇differential(p0\p) + TO.∇²differential(p0\p,Q*err)


# RPs
g = rand(RodriguesParam)
g0 = rand(RodriguesParam)
gz = RodriguesParam(0.,0.,0.)
gval = SVector(g)
ϕ = @SVector zeros(3)

Q = Diagonal(@SVector rand(3))

function mycost(g::SVector{3})
    g = RodriguesParam(g)
    err = SVector(g0\g)
    0.5*err'Q*err
end
function mycostdiff(ϕ::SVector{3})
    mycost(SVector(g*RodriguesParam(ϕ)))
end


grad_d = ForwardDiff.gradient(mycostdiff, ϕ)
hess_d = ForwardDiff.hessian(mycostdiff, ϕ)
err = g ⊖ g0
@test grad_d ≈ TO.∇differential(g0\g)'Q*err
@test hess_d ≈ TO.∇differential(g0\g)'Q*TO.∇differential(g0\g) + TO.∇²differential(g0\g,Q*err)
