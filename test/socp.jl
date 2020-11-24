using StaticArrays, LinearAlgebra
using ForwardDiff
using BenchmarkTools
using Test

function soc_projection(x::StaticVector)
    s = x[end]
    v = pop(x)
    a = norm(v)
    if a <= -s          # below the cone
        return zero(x) 
    elseif a <= s       # in the cone
        return x
    elseif a >= abs(s)  # outside the cone
        return 0.5 * (1 + s/a) * push(v, a)
    else
        throw(ErrorException("Invalid second-order cone projection"))
    end
end

function in_soc(x::StaticVector)
    s = x[end]
    v = pop(x)
    a = norm(v)
    return a <= s
end

function ∇soc_projection(x::StaticVector{n,T}) where {n,T}
    s = x[end]
    v = pop(x)
    a = norm(v)
    if a <= -s
        return @SMatrix zeros(T,n,n)
    elseif a <= s
        return oneunit(SMatrix{n,n,T})
    elseif a >= abs(s)
        b = 0.5 * (1 + s/a)    # scalar
        dbdv = -0.5*s/a^3 * v
        dbds = 0.5 / a
        dvdv = dbdv * v' + b * oneunit(SMatrix{n-1,n-1,T})
        dvds = dbds * v
        dsdv = dbdv * a + b * v / a 
        dsds = dbds * a
        dv = [dvdv dvds]
        ds = push(dsdv, dsds)
        return [dv; ds']
    else
        throw(ErrorException("Invalid second-order cone projection"))
    end
end

function penalty(c, λ)
    inactive = norm(λ, Inf) < 1e-9
    in_cone = in_soc(c)
    if in_cone && !inactive
        return c  # this seems odd to me. Should check if this actually helps
    else
        c_proj = soc_projection(c) 
        return c - c_proj
    end
end

function ∇penalty!(Cbar, C, c::StaticVector{p,T}, λ::StaticVector{p,T}) where {p,n,T}
    s = c[end]
    v = pop(c)
    a = norm(v)
    in_cone = in_soc(c)
    inactive = norm(λ, Inf) < 1e-9
    if in_cone && inactive
        Cbar .*= 0
    elseif in_cone && !inactive
        Cbar .= C
    else  # not in the cone
        if a <= -s  # below the cone
            Cbar .= C
        else        # outside the cone
            ∇proj = I - ∇soc_projection(c)
            mul!(Cbar, ∇proj, C)
        end
    end
end

## Test the methods above
v = @SVector rand(4)
s = norm(v)
J = zeros(5,5)

# inside the cone
x = push(v, s+0.1)
@test soc_projection(x) == x
@test ∇soc_projection(x) ≈ ForwardDiff.jacobian(soc_projection, x) ≈ I(5)
@test ∇soc_projection!(J,x) ≈ I(5)

# outside the code
x = push(v, s-0.1)
@test soc_projection(x) ≈ 0.5*(1 + (s-.1)/norm(v)) * [v; norm(v)]
@test ∇soc_projection(x) ≈ ForwardDiff.jacobian(soc_projection, x)
@test ∇soc_projection!(J,x) ≈ ∇soc_projection(x)
# @btime ∇soc_projection($x)
# @btime ∇soc_projection!($J, $x)
# @btime ForwardDiff.jacobian($soc_projection, $x)

# below the cone
x = push(v, -s-0.1)
@test soc_projection(x) == zero(x)
@test ∇soc_projection(x) ≈ ForwardDiff.jacobian(soc_projection, x) ≈ zeros(5,5)
@test ∇soc_projection!(J,x) ≈ ∇soc_projection(x)


## Penalty term
pen(c) = penalty(c, λ)

c = push(v, s+0.1)  # inside the cone
λ = zero(c)         # multipliers converged
@test pen(c) == zero(c)

C = rand(5,4)
Cbar = zeros(5,4)
@test ∇penalty!(Cbar, C, c, λ) == zeros(5,4)

λ = @SVector rand(5)
@test pen(c) == c
@test ∇penalty!(Cbar, C, c, λ) == C

# Outside the cone
c = push(v, s-0.1)
@test pen(c) ≈ c - soc_projection(c)

@test ∇penalty!(Cbar, C, c, λ) ≈ ForwardDiff.jacobian(pen, c)*C
@test ∇penalty!(Cbar, C, c, λ) ≈ (I - ∇soc_projection(c))*C

# Below the Cone
c = push(v, -s-0.1)
@test pen(c) ≈ c - soc_projection(c) ≈ c
@test ∇penalty!(Cbar, C, c, λ) ≈ ForwardDiff.jacobian(pen, c)*C
@test ∇penalty!(Cbar, C, c, λ) ≈ C 


## Test methods built into TrajOpt
v = @SVector rand(4)
s = norm(v)

function test_soc(v, s)
    SOC = TO.SecondOrderCone()
    n = length(v) + 1
    J = zeros(n,n)
    x = push(v, s+0.1)
    @test TO.projection(SOC, x) == soc_projection(x) == x
    @test TO.∇projection!(SOC, J, x) ≈ ∇soc_projection(x)
    @test (@allocated TO.projection(TO.SecondOrderCone(), x)) == 0
    @test (@allocated TO.∇projection!(SOC, J, x)) == 0
    x = push(v, s-0.1) 
    @test TO.projection(SOC, x) == soc_projection(x)
    @test TO.∇projection!(SOC, J, x) ≈ ∇soc_projection(x)
    @test (@allocated TO.projection(TO.SecondOrderCone(), x)) == 0
    @test (@allocated TO.∇projection!(SOC, J, x)) == 0
    x = push(v, -s-0.1)
    @test TO.projection(SOC, x) == soc_projection(x) == zero(x)
    @test TO.∇projection!(SOC, J, x) ≈ ∇soc_projection(x)
    @test (@allocated TO.projection(TO.SecondOrderCone(), x)) == 0
    @test (@allocated TO.∇projection!(SOC, J, x)) == 0
end
test_soc(v, s)

## Test NormConstraint w/ SOC
#  ||u|| <= 4.2
s = 4.2
n,m = 3,2
x = @SVector rand(n)
u = @SVector rand(m)
z = KnotPoint(x,u,0.1)

normcon1 = NormConstraint(n, m, s, Inequality(), :control)
@test TO.evaluate(normcon1, z) == [u'u - s]
J = zeros(1,n+m)
TO.jacobian!(J, normcon1, z)
@test J ≈ [zero(x); 2u]'
@test length(normcon1) == 1

normcon2 = NormConstraint(n, m, s, TO.SecondOrderCone(), :control)
@test TO.evaluate(normcon2, z) ≈ [u; s]
J = zeros(m+1,n+m)
@test TO.jacobian!(J, normcon2, z) == true  # the Jacobian in constant
@test J ≈ [zeros(m,n) I(m); zeros(1,n+m)]
eval_con2(x) = TO.evaluate(normcon2, StaticKnotPoint(z, x))
@test eval_con2(z.z) ≈ [u; s]
@test ForwardDiff.jacobian(eval_con2, z.z) ≈ J
@test length(normcon2) == m + 1
@test TO.sense(normcon2) == TO.SecondOrderCone()