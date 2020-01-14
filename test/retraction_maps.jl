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

ForwardDiff.jacobian(x->SVector(ModifiedRodriguesParam(x)),p) ≈
    jacobian(ModifiedRodriguesParam, p)

@btime jacobian(ModifiedRodriguesParam, $p)


# Gibbs Vectors
g = @SVector rand(3)
ForwardDiff.jacobian(x->SVector(CayleyMap(x)),g) ≈
    jacobian(CayleyMap, g)

@btime jacobian(CayleyMap, $p)


# Vector Part
v = 0.1*@SVector rand(3)
ForwardDiff.jacobian(x->SVector(VectorPart(x)),v) ≈
    jacobian(VectorPart, v)
@btime jacobian(VectorPart, $v)



jac_eye = [@SMatrix zeros(1,3); Diagonal(@SVector ones(3))];
jacobian(ExponentialMap, p*1e-10) ≈ jac_eye
jacobian(ModifiedRodriguesParam, p*1e-10) ≈ jac_eye
jacobian(CayleyMap, p*1e-10) ≈ jac_eye
jacobian(VectorPart, p*1e-10) ≈ jac_eye
