""""
Specifies whether the constraint is an equality or inequality constraint.
Valid subtypes are `Equality`, `Inequality` ⟺ `NegativeOrthant`, and `SecondOrderCone`.

The sense of a constraint can be queried using `sense(::AbstractConstraint)`

If `sense(con) <: Conic` (i.e. not `Equality`), then the following operations are supported:
* `Base.in(::Conic, x::StaticVector)`. i.e. `x ∈ cone`
* `projection(::Conic, x::StaticVector)`
* `∇projection(::Conic, J, x::StaticVector)`
* `∇²projection(::Conic, J, x::StaticVector, b::StaticVector)`
"""
abstract type ConstraintSense end
abstract type Conic <: ConstraintSense end

"""
Equality constraints of the form ``g(x) = 0`.
Type singleton, so it is created with `Equality()`.
"""
struct Equality <: ConstraintSense end
"""
Inequality constraints of the form ``h(x) \\leq 0``.
Type singleton, so it is created with `Inequality()`. 
Equivalent to `NegativeOrthant`.
"""
struct NegativeOrthant <: Conic end
const Inequality = NegativeOrthant

struct PositiveOrthant <: Conic end 

"""
The second-order cone is defined as 
``\\|x\\| \\leq t``
where ``x`` and ``t`` are both part of the cone.
TrajectoryOptimization assumes the scalar part ``t`` is 
the last element in the vector.
"""
struct SecondOrderCone <: Conic end

dualcone(::NegativeOrthant) = NegativeOrthant()
dualcone(::PositiveOrthant) = PositiveOrthant()
dualcone(::SecondOrderCone) = SecondOrderCone()

projection(::NegativeOrthant, x) = min.(0, x)
projection(::PositiveOrthant, x) = max.(0, x)

@generated function projection(::SecondOrderCone, x::V) where V <: AbstractVector
    # assumes x is stacked [v; s] such that ||v||₂ ≤ s
    v = V <: StaticVector ? :(v = pop(x)) : :(v = view(x, 1:n-1))
    quote
        n = length(x)
        s = x[end]
        $v
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
end

projection!(::Equality, px, x) = px .= 0 

function projection!(::NegativeOrthant, px, x)
    @assert length(px) == length(x)
    for i in eachindex(x)
        px[i] = min(0, x[i])
    end
    return px
end

function projection!(::SecondOrderCone, px, x::V) where V <: AbstractVector
    # assumes x is stacked [v; s] such that ||v||₂ ≤ s
    n = length(x)
    s = x[end]
    v = view(x,1:n-1)
    pv = view(px,1:n-1)
    a = norm(v)
    if a <= -s          # below the cone
        px .= 0
    elseif a <= s       # in the cone
        px .= x
    elseif a >= abs(s)  # outside the cone
        pv .= v
        px[end] = a
        get_data(px) .*= 0.5 * (1 + s/a)
    else
        throw(ErrorException("Invalid second-order cone projection"))
    end
    return pv
end

function ∇projection!(::NegativeOrthant, J, x)
    for i in eachindex(x)
        J[i,i] = x[i] <= 0 ? 1 : 0
    end
    return J
end

@generated function ∇projection!(::SecondOrderCone, J, x::V) where V <: AbstractVector
    v = V <: StaticVector ? :(v = pop(x)) : :(v = view(x, 1:n-1))
    return quote
        n = length(x)
        s = x[end]
        $v
        a = norm(v)
        if a <= -s                               # below cone
            get_data(J) .*= 0
        elseif a <= s                            # in cone
            get_data(J) .*= 0
            for i = 1:n
                J[i,i] = 1.0
            end
        elseif a >= abs(s)                       # outside cone
            # scalar
            c = 0.5 * (1 + s/a)   

            # dvdv = dbdv * v' + c * oneunit(SMatrix{n-1,n-1,T})
            for i = 1:n-1, j = 1:n-1
                J[i,j] = -0.5*s/a^3 * v[i] * v[j]
                if i == j
                    J[i,j] += c
                end
            end

            # dvds
            for i = 1:n-1
                J[i,n] = 0.5 * v[i] / a
            end

            # ds
            for i = 1:n-1
                J[n,i] = ((-0.5*s/a^2) + c/a) * v[i]
            end
            J[n,n] = 0.5 
        else
            throw(ErrorException("Invalid second-order cone projection"))
        end
        return J
    end
end

Base.in(x, ::NegativeOrthant) = all(x->x<=0, x)

function Base.in(x, ::SecondOrderCone)
    s = x[end]
    v = pop(x)
    a = norm(v)
    return a <= s
end

function ∇²projection!(::NegativeOrthant, hess, x, b)
    get_data(hess) .= 0
end

@generated function ∇²projection!(
    ::SecondOrderCone, hess, x::V1, b::V2
) where {V1<:AbstractVecOrMat,V2<:AbstractVector}
    v = V1 <: StaticVector ? :(v = pop(x)) : :(v = view(x, 1:n))
    bv = V2 <: StaticVector ? :(bv = pop(b)) : :(bv = view(b, 1:n))
    quote
        n = length(x)-1
        @assert size(hess) == (n+1,n+1)
        s = x[end]
        $v
        bs = b[end]
        $bv
        a =  norm(v)
        vbv = dot(v,bv)

        if a <= -s
            return get_data(hess) .= 0
        elseif a <= s
            return get_data(hess) .= 0
        elseif a > abs(s)
            # Original equations from chain rule
            # n = n + 1
            # dvdv = -s/norm(v)^2/norm(v)*(I - (v*v')/(v'v))*bv*v' + 
            #     s/norm(v)*((v*(v'bv))/(v'v)^2 * 2v' - (I*(v'bv) + v*bv')/(v'v)) + 
            #     bs/norm(v)*(I - (v*v')/(v'v))
            # dvds = 1/norm(v)*(I - (v*v')/(v'v))*bv;
            # hess[1:n-1,1:n-1] .= dvdv*0.5
            # hess[1:n-1,n] .= dvds*0.5
            # hess[n:n,1:n-1] .= 0.5*dvds'
            # hess[n,n] = 0
            # return hess

            # The following is just an unrolled version of the above
            dvdv = view(hess, 1:n, 1:n)
            dvds = view(hess, 1:n, n+1)
            dsdv = view(hess, n+1, 1:n)
            @inbounds for i = 1:n
                hi = 0
                @inbounds for j = 1:n
                    Hij = -v[i]*v[j] / a^2
                    if i == j
                        Hij += 1
                    end
                    hi += Hij * bv[j]
                end
                dvds[i] = hi / 2a
                dsdv[i] = dvds[i]
                @inbounds for j = 1:i
                    vij = v[i] * v[j]
                    H1 = hi * v[j] * (-s/a^3)
                    H2 = vij * (2*vbv) / a^4 - v[i] * bv[j] / a^2
                    H3 = -vij / a^2
                    if i == j
                        H2 -= vbv / a^2
                        H3 += 1
                    end
                    H2 *= s/a
                    H3 *= bs/a
                    dvdv[i,j] = (H1 + H2 + H3) / 2
                    dvdv[j,i] = dvdv[i,j]
                end
            end
            hess[end,end] = 0
            return hess
        else
            throw(ErrorException("Invalid second-order cone projection"))
        end
    end
end

function cone_status(::SecondOrderCone, x)
    s = x[end]
    v = pop(x)
    a = norm(v)
    if a <= -s
        return :below
    elseif a <= s
        return :in
    elseif a > abs(s)
        return :outside
    else
        return :invalid
    end
end