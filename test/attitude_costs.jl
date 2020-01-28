
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
function rmap(p::SVector{3})
    p /= 2
    n2 = p'p
    M = 2/(1+n2)
    return @SVector [(1-n2)/(1+n2), M*p[1], M*p[2], M*p[3]]
end

function ∇rmap(p::SVector{3})
    μ = 0.5
    μ2 = 0.25
    n = 1+μ2*p'p
    2*[-2*μ2*p'; I*μ*n - 2*μ*μ2*p*p']/n^2
end

function ∇rmap2(p::SVector{3}, b::SVector{4})
    μ = 0.5
    μ2 = 0.25
    n = 1+μ2*p'p
    dn = 2μ2*p'
    v = @SVector [b[2], b[3], b[4]]
    # (-p*b[1] + I*n*v - 0.5*p*p'v)/n^2
    (-I*b[1] + I*v*dn - 0.5*I*(p'v) - 0.5*p*v')/n^2 +
        -2(-p*b[1] + I*n*v - 0.5*p*p'v)/n^3 * dn
end

function imap(q::SVector{4})
    si = 1/(1+q[1])
    return 2*@SVector [q[2]*si, q[3]*si, q[4]*si]
end

function ∇imap(q::SVector{4})
    v = @SVector [q[2], q[3], q[4]]
    si = 1/(1+q[1])
    return 2*@SMatrix [
        -si^2*v[1] si 0 0;
        -si^2*v[2] 0 si 0;
        -si^2*v[3] 0 0 si;
    ]
end

function ∇imap2(q::SVector{4}, b::SVector{3})
    μ = 2
    si = 1/(1+q[1])
    v = @SVector [q[2], q[3], q[4]]
    μ * @SMatrix [
        2*si^3*(v'b) -si^2*b[1] -si^2*b[2] -si^2*b[3];
       -si^2*b[1] 0 0 0;
       -si^2*b[2] 0 0 0;
       -si^2*b[3] 0 0 0;
    ]
end

p = @SVector rand(3)
q = rmap(p)
q0 = normalize(@SVector randn(4))
ϕ = @SVector zeros(3)

ForwardDiff.jacobian(rmap,p) ≈ ∇rmap(p)
ForwardDiff.jacobian(imap,q) ≈ ∇imap(q)
b = @SVector rand(3)
ForwardDiff.jacobian(x->∇imap(x)'b, q) ≈ ∇imap2(q,b)
b = @SVector rand(4)
ForwardDiff.jacobian(x->∇rmap(x)'b, p) ≈ ∇rmap2(p,b)

function mycost(q::SVector{4})
    dq = Lmult(q0)'q
    0.5*imap(dq)'imap(dq)
end
function mycostdiff(ϕ::SVector{3})
    mycost(Lmult(q)*rmap(ϕ))
end
dq = Lmult(q0)'q
grad_q = ForwardDiff.gradient(mycost, q)
hess_q = ForwardDiff.hessian(mycost, q)
grad_q ≈ Lmult(q0)*∇imap(dq)'imap(dq)
hess_q ≈ Lmult(q0)*∇imap2(dq,imap(dq))*Lmult(q0)' + Lmult(q0)*∇imap(dq)'∇imap(dq)*Lmult(q0)'

grad_p = ForwardDiff.gradient(mycostdiff, ϕ)
hess_p = ForwardDiff.hessian(mycostdiff, ϕ)

grad_p ≈ ∇rmap(ϕ)'Lmult(q)'*Lmult(q0)*∇imap(dq)'imap(dq)
b = Lmult(q)'*Lmult(q0)*∇imap(dq)'imap(dq)
b ≈ Lmult(q)'grad_q
hess_p ≈ ∇rmap2(ϕ,b) + ∇rmap(ϕ)'Lmult(q)'Lmult(q0)*∇imap2(dq,imap(dq))*Lmult(q0)'Lmult(q)*∇rmap(ϕ) +
    ∇rmap(ϕ)'Lmult(q)'*Lmult(q0)*∇imap(dq)'∇imap(dq)*Lmult(q0)'Lmult(q)*∇rmap(ϕ)

b = @SVector rand(4)
∇rmap2(p,b) ≈ ∇rmap(p)'b

ForwardDiff.jacobian(mycost, q) ≈ ∇imap(q)


err'∇imap(dq)*Lmult(q0)'
