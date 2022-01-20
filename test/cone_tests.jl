using Test
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using ForwardDiff
const TO = TrajectoryOptimization

function Πsoc(x)
    v = x[1:end-1]
    s = x[end]
    a = norm(v)
    if a <= -s
        return zero(x)
    elseif a <= s
        return x
    elseif a >= abs(s)
        x̄ = append!(v, a)
        return 0.5*(1 + s/a) * x̄
    end
    throw(ErrorException("Invalid second-order cone"))
end

Πineq(x) = min.(0, x)

function testcone(cone, x)
    px = similar(x)
    n = length(x) - 1
    J = zeros(n+1,n+1)
    H = zeros(n+1,n+1)
    b = similar(x)
    b .= randn(n+1)

    TO.projection!(cone, px, x)
    if cone isa TO.SecondOrderCone
        Π = Πsoc
    else
        Π = Πineq
    end
    @test px ≈ Π(x)
    TO.∇projection!(cone, J, x)
    TO.∇²projection!(cone, H, x, b)
    @test J ≈ ForwardDiff.jacobian(Π, x)
    @test H ≈ ForwardDiff.hessian(x->Π(x)'b, x)

    @test (@allocated TO.∇projection!(cone, J, x)) == 0
    @test (@allocated TO.∇²projection!(cone, H, x, b)) == 0
end

cone = TO.SecondOrderCone()
n = 3
u = SA[2,3,1.] 
x = push(u, 1.0)
@test x ∉ cone
@test TO.cone_status(cone, x) == :outside
testcone(cone, x)
testcone(cone, Vector(x))

x = SA[2,3,1,-10.]
@test x ∉ cone
@test TO.cone_status(cone, x) == :below
testcone(cone, x)
testcone(cone, Vector(x))

x = SA[2,3,1,10.]
@test TO.cone_status(cone, x) == :in
@test x ∈ cone
testcone(cone, x)
testcone(cone, Vector(x))

# Inequality 
cone = TO.Inequality()
x = SA[1,2,-3.]
@test x ∉ cone
testcone(cone, x)
testcone(cone, Vector(x))
