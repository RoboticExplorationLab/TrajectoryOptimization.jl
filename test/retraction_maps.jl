# Exponential
ϕ = @SVector rand(3)
ExponentialMap(ϕ)
ForwardDiff.jacobian(x->SVector(ExponentialMap(x)),ϕ) ≈ jacobian(ExponentialMap,ϕ)
@btime jacobian(ExponentialMap,$ϕ)

ϕ = 1e-6*@SVector rand(3)
ForwardDiff.jacobian(x->SVector(ExponentialMap(x)),ϕ) ≈ jacobian(ExponentialMap,ϕ)
@btime jacobian(ExponentialMap,$(ϕ*1e-6))



# MRPs
p = SVector(rand(MRP{Float64}))

ForwardDiff.jacobian(x->SVector(MRPMap(x)),p) ≈
    jacobian(MRPMap, p)

@btime jacobian(MRPMap, $p)


# Gibbs Vectors
g = @SVector rand(3)
ForwardDiff.jacobian(x->SVector(CayleyMap(x)),g) ≈ jacobian(CayleyMap, g)
@btime jacobian(CayleyMap, $p)


# Vector Part
v = 0.1*@SVector rand(3)
ForwardDiff.jacobian(x->SVector(VectorPart(x)),v) ≈
    jacobian(VectorPart, v)
@btime jacobian(VectorPart, $v)



jac_eye = [@SMatrix zeros(1,3); 0.5*Diagonal(@SVector ones(3))];
jacobian(ExponentialMap, p*1e-10) ≈ jac_eye
jacobian(MRPMap, p*1e-10) ≈ jac_eye
jacobian(CayleyMap, p*1e-10) ≈ jac_eye
jacobian(VectorPart, p*1e-10) ≈ jac_eye


############################################################################################
#                                 INVERSE RETRACTION MAPS
############################################################################################

# Exponential Map
q = rand(UnitQuaternion{Float64})
q = UnitQuaternion{ExponentialMap}(q)
qval = SVector(q)
ExponentialMap(q) == logm(q)
ExponentialMap(ExponentialMap(q)) ≈ q
ExponentialMap(ExponentialMap(ϕ)) ≈ ϕ

function invmap(q)
    v = @SVector [q[2], q[3], q[4]]
    s = q[1]
    θ = norm(v)
    M = 2atan(θ, s)/θ
    return M*v
end
invmap(qval) ≈ logm(q)

ForwardDiff.jacobian(invmap, qval) ≈ jacobian(ExponentialMap, q)

# Vector Part
VectorPart(q) == 2*qval[2:4]
jacobian(VectorPart, q)
VectorPart(VectorPart(q)) ≈ q
VectorPart(VectorPart(v)) ≈ v

# Cayley
invmap(q) = 1/q[1] * 2*@SVector [q[2], q[3], q[4]]
CayleyMap(q) ≈ invmap(qval)
ForwardDiff.jacobian(invmap, qval) ≈ jacobian(CayleyMap, q)
CayleyMap(CayleyMap(q)) ≈ q
CayleyMap(CayleyMap(g)) ≈ g

# MRP
invmap(q) = 4/(1+q[1]) * @SVector [q[2], q[3], q[4]]
MRPMap(q) ≈ invmap(qval)
ForwardDiff.jacobian(invmap, qval) ≈ jacobian(MRPMap, q)
MRPMap(MRPMap(q)) ≈ q
MRPMap(MRPMap(p)) ≈ p
