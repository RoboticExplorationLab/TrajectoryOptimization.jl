import Base: +, -, *, /, \, exp, log, ≈, inv, conj
import LinearAlgebra: norm2

export
    Rotation,
    UnitQuaternion,
    MRP,
    RPY,
    RodriguesParam,
    ExponentialMap,
    VectorPart,
    MRPMap,
    CayleyMap,
    IdentityMap,
    ReNorm

export
    differential_rotation,
    retraction_map,
    scalar,
    vector,
    logm,
    expm,
    kinematics,
    rotmat,
    Lmult,
    Rmult,
    Vmat,
    Hmat,
    Tmat,
    skew,
    vee,
    ⊕,
    ⊖


function skew(v::AbstractVector)
    @assert length(v) == 3
    @SMatrix [0   -v[3]  v[2];
              v[3] 0    -v[1];
             -v[2] v[1]  0]
end

function vee(S::AbstractMatrix)
    return @SVector [S[3,2], S[1,3], S[2,1]]
end

abstract type Rotation end

abstract type DifferentialRotation end
abstract type VectorPart <: DifferentialRotation end
abstract type ExponentialMap <: DifferentialRotation end
abstract type MRPMap <: DifferentialRotation end
abstract type CayleyMap <: DifferentialRotation end
abstract type IdentityMap <: DifferentialRotation end
abstract type ReNorm <: DifferentialRotation end

# Scalings
@inline scaling(::Type{ExponentialMap}) = 0.5
@inline scaling(::Type{VectorPart}) = 1.0
@inline scaling(::Type{CayleyMap}) = 1.0
@inline scaling(::Type{MRPMap}) = 2.0

# Retraction Maps
(::Type{ExponentialMap})(ϕ) = expm(ϕ/scaling(ExponentialMap))

function (::Type{VectorPart})(v)
    μ = 1/scaling(VectorPart)
    UnitQuaternion{VectorPart}(sqrt(1-μ^2*v'v), μ*v[1], μ*v[2], μ*v[3])
end

function (::Type{CayleyMap})(g)
    g /= scaling(CayleyMap)
    M = 1/sqrt(1+g'g)
    UnitQuaternion{CayleyMap}(M, M*g[1], M*g[2], M*g[3])
end

function (::Type{MRPMap})(p)
    p /= scaling(MRPMap)
    n2 = p'p
    M = 2/(1+n2)
    UnitQuaternion{MRPMap}((1-n2)/(1+n2), M*p[1], M*p[2], M*p[3])
end

(::Type{IdentityMap})(q) = UnitQuaternion{IdentityMap}(q[1], q[2], q[3], q[4])

# Retraction Map Jacobians
function jacobian(::Type{ExponentialMap},ϕ, eps=1e-5)
    μ = 1/scaling(ExponentialMap)
    θ = norm(ϕ)
    cθ = cos(μ*θ/2)
    sincθ = sinc(μ*θ/2π)
    if θ < eps
        0.5*μ*[-0.5*μ*sincθ*ϕ'; sincθ*I + (cθ - sincθ)*ϕ*ϕ']
    else
        0.5*μ*[-0.5*μ*sincθ*ϕ'; sincθ*I + (cθ - sincθ)*ϕ*ϕ'/(ϕ'ϕ)]
    end
end

function jacobian(::Type{VectorPart}, v)
    μ = 1/scaling(VectorPart)
    μ2 = μ*μ
    M = -μ2/sqrt(1-μ2*v'v)
    @SMatrix [v[1]*M v[2]*M v[3]*M;
              μ 0 0;
              0 μ 0;
              0 0 μ]
end

function jacobian(::Type{CayleyMap}, g)
    μ = 1/scaling(CayleyMap)
    μ2 = μ*μ
    n = 1+μ2*g'g
    ni = 1/n
    μ*[-μ*g'; -μ2*g*g' + I*n]*ni*sqrt(ni)
end

function jacobian(::Type{MRPMap}, p)
    μ = 1/scaling(MRPMap)
    μ2 = μ*μ
    n = 1+μ2*p'p
    2*[-2*μ2*p'; I*μ*n - 2*μ*μ2*p*p']/n^2
end

jacobian(::Type{IdentityMap}, q) = I


const DEFAULT_QUATDIFF = VectorPart

""" $(TYPEDEF)
4-parameter attitute representation that is singularity-free. Quaternions with unit norm
represent a double-cover of SO(3). The `UnitQuaternion` does NOT strictly enforce the unit
norm constraint, but certain methods will assume you have a unit quaternion. The
`UnitQuaternion` type is parameterized by the linearization method, which maps quaternions
to the 3D plane tangent to the 4D unit sphere. Follows the Hamilton convention for quaternions.

There are currently 4 methods supported:
* `VectorPart` - uses the vector (or imaginary) part of the quaternion
* `ExponentialMap` - the most common approach, uses the exponential and logarithmic maps
* `CayleyMap` - or Rodrigues parameters (aka Gibbs vectors).
* `MRPMap` - or Modified Rodrigues Parameter, is a sterographic projection of the 4D unit sphere
onto the plane tangent to either the positive or negative real poles.

# Constructors
```julia
UnitQuaternion(s,x,y,z)  # defaults to `VectorPart`
UnitQuaternion{D}(s,x,y,z)
UnitQuaternion{D}(q::SVector{4})
UnitQuaternion{D}(r::SVector{3})  # quaternion with 0 real part
```
"""
struct UnitQuaternion{T,D<:DifferentialRotation} <: Rotation
    s::T
    x::T
    y::T
    z::T
end

UnitQuaternion(s::T,x::T,y::T,z::T) where T = UnitQuaternion{T,DEFAULT_QUATDIFF}(s,x,y,z)
UnitQuaternion(q::SVector{4}) = UnitQuaternion{DEFAULT_QUATDIFF}(q[1],q[2],q[3],q[4])
UnitQuaternion{D}(s::T,x::T,y::T,z::T) where {T,D} = UnitQuaternion{T,D}(s,x,y,z)

UnitQuaternion{D}(q::SVector{4}) where D = UnitQuaternion{D}(q[1],q[2],q[3],q[4])
UnitQuaternion{D}(r::SVector{3}) where D = UnitQuaternion{D}(0.0, r[1],r[2],r[3])
UnitQuaternion{D}(q::UnitQuaternion) where D = UnitQuaternion{D}(q.s, q.x, q.y, q.z)
UnitQuaternion{T,D}(q::R) where {T,D,R <: UnitQuaternion} =
    UnitQuaternion{T,D}(q.s, q.x, q.y, q.z)

UnitQuaternion(r::SVector{3}) = UnitQuaternion{DEFAULT_QUATDIFF}(0.0, r[1],r[2],r[3])
UnitQuaternion(q::UnitQuaternion) = q

(::Type{UnitQuaternion{T,D}})(x::SVector{4,T2}) where {T,T2,D} =
    UnitQuaternion{promote_type(T,T2),D}(x[1], x[2], x[3], x[4])

retraction_map(::UnitQuaternion{T,D}) where {T,D} = D
retraction_map(::Type{UnitQuaternion{T,D}}) where {T,D} = D

Base.rand(::Type{<:UnitQuaternion{T,D}}) where {T,D} =
    normalize(UnitQuaternion{T,D}(randn(T), randn(T), randn(T), randn(T)))

Base.rand(::Type{UnitQuaternion{T}}) where T = Base.rand(UnitQuaternion{T,VectorPart})
Base.rand(::Type{UnitQuaternion}) = Base.rand(UnitQuaternion{Float64,VectorPart})
Base.zero(::Type{Q}) where Q<:UnitQuaternion = I(Q)
Base.zero(q::Q) where Q<:UnitQuaternion = I(Q)

SVector(q::UnitQuaternion{T}) where T = SVector{4,T}(q.s, q.x, q.y, q.z)

scalar(q::UnitQuaternion) = q.s
vector(q::UnitQuaternion{T}) where T = SVector{3,T}(q.x, q.y, q.z)
vecnorm(q::UnitQuaternion) = sqrt(q.x^2 + q.y^2 + q.z^2)

conj(q::UnitQuaternion{T,D}) where {T,D} = UnitQuaternion{T,D}(q.s, -q.x, -q.y, -q.z)
inv(q::UnitQuaternion) = conj(q)

LinearAlgebra.norm2(q::UnitQuaternion) = q.s^2 + q.x^2 + q.y^2 + q.z^2
LinearAlgebra.norm(q::UnitQuaternion) = sqrt(q.s^2 + q.x^2 + q.y^2 + q.z^2)
LinearAlgebra.I(::Type{UnitQuaternion}) = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
LinearAlgebra.I(::Type{Q}) where Q <: UnitQuaternion = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
(::Type{Q})(I::UniformScaling) where Q<:UnitQuaternion = Q(1.0, 0.0, 0.0, 0.0)
(≈)(q::UnitQuaternion, u::UnitQuaternion) = q.s ≈ u.s && q.x ≈ u.x && q.y ≈ u.y && q.z ≈ u.z
(-)(q::UnitQuaternion{T,D}) where {T,D} = UnitQuaternion{T,D}(-q.s, -q.x, -q.y, -q.z)

function LinearAlgebra.normalize(q::UnitQuaternion{T,D}) where {T,D}
    n = norm(q)
    UnitQuaternion{T,D}(q.s/n, q.x/n, q.y/n, q.z/n)
end

function (*)(q::UnitQuaternion{T1,D1}, w::UnitQuaternion{T2,D2}) where {T1,T2,D1,D2}
    T = promote_type(T1, T2)
    D = D2
    UnitQuaternion{T,D}(q.s * w.s - q.x * w.x - q.y * w.y - q.z * w.z,
                        q.s * w.x + q.x * w.s + q.y * w.z - q.z * w.y,
                        q.s * w.y - q.x * w.z + q.y * w.s + q.z * w.x,
                        q.s * w.z + q.x * w.y - q.y * w.x + q.z * w.s)
end

function Base.:*(q::UnitQuaternion{Tq}, r::SVector{3}) where Tq
    qo = (-q.x  * r[1] - q.y * r[2] - q.z * r[3],
           q.s  * r[1] + q.y * r[3] - q.z * r[2],
           q.s  * r[2] - q.x * r[3] + q.z * r[1],
           q.s  * r[3] + q.x * r[2] - q.y * r[1])

   T = promote_type(Tq, eltype(r))

   return similar_type(r, T)(-qo[1] * q.x + qo[2] * q.s - qo[3] * q.z + qo[4] * q.y,
                             -qo[1] * q.y + qo[2] * q.z + qo[3] * q.s - qo[4] * q.x,
                             -qo[1] * q.z - qo[2] * q.y + qo[3] * q.x + qo[4] * q.s)
end

function (*)(q::Q, s::Real) where Q<:UnitQuaternion
    return Q(q.s*s, q.x*s, q.y*s, q.z*s)
end

(\)(q1::UnitQuaternion, q2::UnitQuaternion) = inv(q1)*q2
(/)(q1::UnitQuaternion, q2::UnitQuaternion) = q1*inv(q2)

(\)(q::UnitQuaternion, r::SVector{3}) = inv(q)*r

function exp(q::UnitQuaternion{T,D}) where {T,D}
    θ = vecnorm(q)
    sθ,cθ = sincos(θ)
    es = exp(q.s)
    M = es*sθ/θ
    UnitQuaternion{T,D}(es*cθ, q.x*M, q.y*M, q.z*M)
end

function expm(ϕ::SVector{3,T}) where T
    θ = norm(ϕ)
    sθ,cθ = sincos(θ/2)
    M = 0.5*sinc(θ/2π)
    UnitQuaternion{T,ExponentialMap}(cθ, ϕ[1]*M, ϕ[2]*M, ϕ[3]*M)
end

# Assumes unit quaternion
function log(q::UnitQuaternion{T,D}, eps=1e-6) where {T,D}
    θ = vecnorm(q)
    if θ > eps
        M = atan(θ, q.s)/θ
    else
        M = (1-(θ^2/(3q.s^2)))/q.s
    end
    UnitQuaternion{T,D}(0.0, q.x*M, q.y*M, q.z*M)
end

# Assumes unit quaternion
function logm(q::UnitQuaternion{T}) where T
    q = log(q)
    SVector{3,T}(2*q.x, 2*q.y, 2*q.z)
end

function rotmat(q::UnitQuaternion)
    s = q.s
    v = vector(q)
    (s^2 - v'v)*I + 2*v*v' + 2*s*skew(v)
end

function kinematics(q::UnitQuaternion{T,D}, ω::SVector{3}) where {T,D}
    SVector(q*UnitQuaternion{T,D}(0.0, 0.5*ω[1], 0.5*ω[2], 0.5*ω[3]))
end

function Base.angle(q::UnitQuaternion)
    min(2*atan(vecnorm(q), q.s), 2*atan(vecnorm(q), -q.s))
end

function (⊕)(q::UnitQuaternion{T,ExponentialMap}, δq::SVector{3}) where T
    q*expm(δq)
end

function (⊕)(q::UnitQuaternion{T,VectorPart}, δq::SVector{3}) where T
    q*UnitQuaternion{VectorPart}(sqrt(1 - δq[1]^2 - δq[2]^2 - δq[3]^2),
        δq[1], δq[2], δq[3])
end

function (⊖)(q::UnitQuaternion{T,D}, q0::UnitQuaternion) where {T,D}
    D(q0\q)
end

function (⊖)(q::UnitQuaternion{T,IdentityMap}, q0::UnitQuaternion) where {T}
    SVector(q) - SVector(q0)
    # return SVector(q0\q)
end

"""
Jacobian of q ⊕ ϕ, when ϕ is near zero. Useful for converting Jacobians from R⁴ to R³ and
    correctly account for unit norm constraint. Jacobians for different
    differential quaternion parameterization are the same up to a constant.
"""
function ∇differential(q::UnitQuaternion)
    1.0 * @SMatrix [
        -q.x -q.y -q.z;
         q.s -q.z  q.y;
         q.z  q.s -q.x;
        -q.y  q.x  q.s;
    ]
end

"Lmult(q2)q1 returns a vector equivalent to q2*q1 (quaternion multiplication)"
function Lmult(q::UnitQuaternion)
    @SMatrix [
        q.s -q.x -q.y -q.z;
        q.x  q.s -q.z  q.y;
        q.y  q.z  q.s -q.x;
        q.z -q.y  q.x  q.s;
    ]
end
Lmult(q::SVector{4}) = Lmult(UnitQuaternion(q))

"Rmult(q1)q2 return a vector equivalent to q2*q1 (quaternion multiplication)"
function Rmult(q::UnitQuaternion)
    @SMatrix [
        q.s -q.x -q.y -q.z;
        q.x  q.s  q.z -q.y;
        q.y -q.z  q.s  q.x;
        q.z  q.y -q.x  q.s;
    ]
end
Rmult(q::SVector{4}) = Rmult(UnitQuaternion(q))

"Tmat()q return a vector equivalent to inv(q)"
function Tmat()
    @SMatrix [
        1  0  0  0;
        0 -1  0  0;
        0  0 -1  0;
        0  0  0 -1;
    ]
end

"Vmat(q)q returns the imaginary (vector) part of the quaternion q (equivalent to vector(q))"
function Vmat()
    @SMatrix [
        0 1 0 0;
        0 0 1 0;
        0 0 0 1
    ]
end

function Hmat()
    @SMatrix [
        0 0 0;
        1 0 0;
        0 1 0;
        0 0 1.;
    ]
end

function Hmat(r)
    @assert length(r) == 3
    @SVector [0,r[1],r[2],r[3]]
end

"Jacobian of q*r with respect to the quaternion"
function ∇rotate(q::UnitQuaternion{T,D}, r::SVector{3}) where {T,D}
    rhat = UnitQuaternion{D}(r)
    R = Rmult(q)
    2Vmat()*Rmult(q)'Rmult(rhat)
end

"Jacobian of q2*q1 with respect to q1"
function ∇composition1(q2::UnitQuaternion, q1::UnitQuaternion)
    Lmult(q2)
end

"Jacobian of q2*q1 with respect to q2"
function ∇composition2(q2::UnitQuaternion, q1::UnitQuaternion)
    Rmult(q1)
end

function ∇²differential(q::UnitQuaternion, b::SVector{4})
    b1 = -SVector(q)'b
    Diagonal(@SVector fill(b1,3))
end


############################################################################################
#                             MODIFIED RODRIGUES PARAMETERS                                #
############################################################################################

""" $(TYPEDEF)
Modified Rodrigues Parameter. Is a 3D parameterization of attitude, and is a sterographic
projection of the 4D unit sphere onto the plane tangent to the negative real pole. They
have a singularity at θ = ±180°.

# Constructors
MRP(x, y, z)
MRP(r::SVector{3})
"""
struct MRP{T} <: Rotation
    x::T
    y::T
    z::T
end

MRP(r::SVector{3}) = MRP(r[1], r[2], r[3])
(::Type{<:MRP})(::Type{T},x,y,z) where T = MRP{T}(T(x),T(y),T(z))
MRP{T}(r::SVector{3}) where T = MRP(T(r[1]), T(r[2]), T(r[3]))
MRP{T}(p::MRP) where T = MRP(T(p.x), T(p.y), T(p.z))
function (::Type{<:MRP})(q::UnitQuaternion)
    M = 1/(1+q.s)
    MRP(q.x*M, q.y*M, q.z*M)
end

function UnitQuaternion(p::MRP)
    n = norm2(p)
    s = (1-n)/(1+n)
    M = 2/(1+n)
    UnitQuaternion{MRPMap}(s, p.x*M, p.y*M, p.z*M)
end

function rotmat(p::MRP)
    p = SVector(p)
    P = skew(p)
    R2 = I + 4( (1-p'p)I + 2*P )*P/(1+p'p)^2
end

SVector(p::MRP{T}) where T = SVector{3,T}(p.x, p.y, p.z)

Base.rand(::Type{MRP{T}}) where T = MRP(rand(UnitQuaternion{T}))
Base.rand(::Type{MRP}) = MRP(rand(UnitQuaternion))
Base.zero(::Type{MRP{T}}) where T = MRP(zero(T), zero(T), zero(T))
Base.zero(::Type{MRP}) = MRP(0.0, 0.0, 0.0)

LinearAlgebra.norm(p::MRP) = sqrt(p.x^2 + p.y^2 + p.z^2)
LinearAlgebra.norm2(p::MRP) = p.x^2 + p.y^2 + p.z^2

(≈)(p2::MRP, p1::MRP) = p2.x ≈ p1.x && p2.y ≈ p1.y && p2.z ≈ p1.z

function (*)(p2::MRP, p1::MRP)
    p2, p1 = SVector(p2), SVector(p1)
    MRP(((1-p2'p2)*p1 + (1-p1'p1)*p2 - cross(2p1, p2) ) / (1+p1'p1*p2'p2 - 2p1'p2))
end

(*)(p::MRP, r::SVector) = UnitQuaternion(p)*r
(\)(p::MRP, r::SVector) = UnitQuaternion(p)\r

function (\)(p1::MRP, p2::MRP)
    # fun fact: equivalent to p2 - p1 when either is 0
    p1,p2 = SVector(p1), SVector(p2)
    n1,n2 = p1'p1, p2'p2
    θ = 1/((1+n1)*(1+n2))
    s1,s2 = (1-n1), (1-n2)
    v1 = -2p1
    v2 =  2p2
    s = s1*s2 - v1'v2
    v = s1*v2 + s2*v1 + v1 × v2

    M = θ/(1+θ*s)
    return MRP(v*M)
end

function ∇err(p1::MRP, p2::MRP)
    n1,n2 = norm2(p1),   norm2(p2)
    θ = 1/((1+n1)*(1+n2))
    s1,s2 = (1-n1), (1-n2)
    p1,p2 = SVector(p1), SVector(p2)
    v1 = -2p1
    v2 =  2p2
    s = s1*s2 - v1'v2
    v = s1*v2 + s2*v1 + v1 × v2
    dsdp = -2s1*p2 - 2v1
    dvdp = 2s1*I + -2v1*p2' + 2skew(v1)
    dθdp = -θ^2*(1+n1)*2p2

    M = θ/(1+θ*s)
    dMdp = 1/(1+θ*s)*dθdp - θ/(1+θ*s)^2*(dθdp*s + θ*dsdp)
    return dvdp*M + v*dMdp'
end

function ∇²err(p1::MRP, p2::MRP, b::SVector{3})
    n1,n2 = norm2(p1),   norm2(p2)
    θ = 1/((1+n1)*(1+n2))
    s1,s2 = (1-n1), (1-n2)
    p1,p2 = SVector(p1), SVector(p2)
    v1 = -2p1
    v2 =  2p2
    s = s1*s2 - v1'v2
    v = s1*v2 + s2*v1 + v1 × v2

    dsdp = -2s1*p2 - 2v1  # 3x1
    dsdp2 = -2s1*I  # 3x3

    dvdp = 2s1*b + -2p2*v1'b - 2skew(v1)*b
    dvdp2 = -I*2v1'b

    dθdp = -θ^2*(1+n1)*2p2  # 3x1
    dθdp2 = -2θ*(1+n1)*2p2*dθdp' - θ^2*(1+n1)*2I # 3x3

    M = θ/(1+θ*s)  # scalar
    dMdp = 1/(1+θ*s)*dθdp - θ/(1+θ*s)^2*(dθdp*s + θ*dsdp) # 3x1
    dM2 = θ/(1+θ*s)^2  # scalar
    dM3 = dθdp*s + θ*dsdp  # 3x1
    dM2dp = dθdp'/(1+θ*s)^2 - 2θ/(1+θ*s)^3 * (dθdp*s + θ*dsdp)'
    dM3dp = dθdp2*s + dθdp*dsdp' + dsdp*dθdp' + θ*dsdp2

    dMdp2 = -1/(1+θ*s)^2*dθdp*(dθdp*s + dsdp*θ)' + 1/(1+θ*s)*dθdp2  # good
    dMdp2 -= dM3*dM2dp + dM2*dM3dp

    vb = s1*v2'b + s2*v1'b + b'skew(v1)*v2  # scalar
    vpdp = s1*2b' - 2p2' * (v1'b)  + b'skew(v1)*2 # good
    # vpdp = s1*2b' - 2p2 * (v1'b)

    d1 = M*dvdp2 + dvdp*dMdp'
    d2 = dMdp2*vb + dMdp*vpdp
    return d1 + d2
end

function (/)(p1::MRP, p2::MRP)
    n1,n2 = norm2(p1),   norm2(p2)
    θ = 1/((1+n1)*(1+n2))
    s1,s2 = (1-n1), (1-n2)
    p1,p2 = SVector(p1), SVector(p2)
    v1 =  2p1
    v2 = -2p2
    s = s1*s2 - v1'v2
    v = s1*v2 + s2*v1 + v1 × v2

    M = θ/(1+θ*s)
    MRP(v[1]*M, v[2]*M, v[3]*M)

end

Base.angle(p::MRP) = angle(UnitQuaternion(p))

function kinematics(p::MRP, ω)
    p = SVector(p)
    A = @SMatrix [
        1 + p[1]^2 - p[2]^2 - p[3]^2  2(p[1]*p[2] - p[3])      2(p[1]*p[3] + p[2]);
        2(p[2]*p[1] + p[3])            1-p[1]^2+p[2]^2-p[3]^2   2(p[2]*p[3] - p[1]);
        2(p[3]*p[1] - p[2])            2(p[3]*p[2] + p[1])      1-p[1]^2-p[2]^2+p[3]^2]
    0.25*A*ω
end

(⊕)(p::MRP, δp::SVector{3}) = p*MRP(δph)
(⊖)(p::MRP, p0::MRP) = SVector(p0\p)


function ∇rotate(p::MRP, r)
    p = SVector(p)
    4( (1-p'p)*skew(r)*(4p*p'/(1+p'p) - I) - (4/(1+p'p)*skew(p) + I)*2*skew(p)*r*p'
      - 2*(skew(p)*skew(r) + skew(skew(p)*r)))/(1+p'p)^2
end

function ∇composition2(p2::MRP, p1::MRP)
    p2,p1 = SVector(p2), SVector(p1)
    n1 = p1'p1
    n2 = p2'p2
    D = 1 / (1+n1*n2 - 2p1'p2)
    d1 = (-2p1*p2' + (1-n1)*I - skew(2p1) ) * D
    d2 = -((1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2) ) * D^2 *
        (2p2*n1 - 2p1)'
    return d1 + d2
end

function ∇composition1(p2::MRP, p1::MRP)
    p2,p1 = SVector(p2), SVector(p1)
    n1 = p1'p1
    n2 = p2'p2
    D = 1 / (1+n1*n2 - 2p1'p2)
    d1 = ((1-n2)*I + -2p2*p1' + 2skew(p2) ) * D
    d2 = -((1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2) ) * D^2 *
        (2p1*n2 - 2p2)'
    d1 + d2
end

function ∇differential(p::MRP)
    p = SVector(p)
    n = p'p
    # p = SVector(p)
    # (1-n)I + 2(skew(p) + p*p')
    # @SMatrix [n + 2p.x^2      2(p.x*p.y-p.z)  2(p.x*p.z+p.y);
    #           2(p.y*p.x+p.z)  n + 2p.y^2      2(p.y*p.z-p.x);
    #           2(p.z*p.x-p.y)  2(p.z*p.y+p.x)  n + 2p.z^2]
    #
    # p2 = SVector(p)
    # n2 = p2'p2
    return (1-n)*I + 2(skew(p) + p*p')
end

function ∇²composition1(p2::MRP, p1::MRP, b::SVector{3})
    p2,p1 = SVector(p2), SVector(p1)
    n1 = p1'p1
    n2 = p2'p2
    D = 1 / (1+n1*n2 - 2p1'p2)  # scalar
    dD = -D^2 * (n2*2p1 - 2p2)  # 3x1
    A = -((1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2) )  # 3x1
    dA = -I*(1-n2) + 2p2*p1' - 2skew(p2)  # 3x3
    B = 2(p1*n2 -  p2)'b  # scalar
    dB = 2n2*b  # 3x1
    d1 = -2p2*b' * D + ((1-n2)*b + -2p2*p1'b + 2skew(p2)*b )*dD'
    d2 = dA * D^2 * B +
         A * 2D * dD' * B +
         A * D^2 * dB'
    return d1 + d2
end

function ∇²differential(p2::MRP, b::SVector{3})
    p2 = SVector(p2)
    n2 = p2'p2
    A = -p2  # 3x1
    B = -2p2  # 3x1
    D = 1
    dD = 2p2

    dA = -I*(1-n2) - 2skew(p2)  # 3x3
    dB = 2n2*I  # 3x3

    d1 = (-2p2'b*I*D) - (dA'b * dD')
    d2 = dB * A'b * D^2 +
        (-(1-n2)*b + (skew(2p2))*b)*B' * D^2 +
        B*A'b * 2D * dD'
    d1 + d2'
end




############################################################################################
#                                 Rodrigues Parameters
############################################################################################

struct RodriguesParam{T} <: Rotation
    x::T
    y::T
    z::T
end

RodriguesParam(g::SVector{3,T}) where T = RodriguesParam{T}(g[1], g[2], g[3])
(::Type{<:RodriguesParam})(::Type{T},x,y,z) where T = RodriguesParam{T}(T(x),T(y),T(z))
(::Type{<:RodriguesParam})(p::RodriguesParam) = p
SVector(g::RodriguesParam{T}) where T = SVector{3,T}(g.x, g.y, g.z)

(::Type{<:RodriguesParam})(q::UnitQuaternion{T}) where T = RodriguesParam(q.x/q.s, q.y/q.s, q.z/q.s)

function UnitQuaternion(g::RodriguesParam{T}) where T
    M = 1/sqrt(1+norm2(g))
    UnitQuaternion{T,CayleyMap}(M, M*g.x, M*g.y, M*g.z)
end

Base.rand(::Type{<:RodriguesParam}) = RodriguesParam(rand(UnitQuaternion))
Base.zero(::Type{<:RodriguesParam}) = RodriguesParam(0.0, 0.0, 0.0)

LinearAlgebra.norm(g::RodriguesParam) = sqrt(g.x^2 + g.y^2 + q.z^2)
LinearAlgebra.norm2(g::RodriguesParam) = g.x^2 + g.y^2 + g.z^2

function (≈)(g2::RodriguesParam, g1::RodriguesParam)
    g2.x ≈ g1.x && g2.y ≈ g1.y && g2.z ≈ g1.z
end

function (*)(g2::RodriguesParam, g1::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)
    RodriguesParam((g2+g1 + g2 × g1)/(1-g2'g1))
end

(*)(g::RodriguesParam, r::SVector{3}) = UnitQuaternion(g)*r
(\)(g::RodriguesParam, r::SVector{3}) = inv(UnitQuaternion(g))*r

"Same as inv(g1)*g2"
function (\)(g1::RodriguesParam, g2::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)
    RodriguesParam((g2-g1 + g2 × g1)/(1+g1'g2))
end

"Same as g2*inv(g1)"
function (/)(g1::RodriguesParam, g2::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)
    RodriguesParam((g1-g2 + g2 × g1)/(1+g1'g2))
end

function kinematics(g::RodriguesParam, ω::SVector{3})
    g = SVector(g)
    0.5*(I + skew(g) + g*g')*ω
end

(⊕)(g::RodriguesParam, δg::SVector{3}) = g*RodriguesParam(δg)
(⊖)(g::RodriguesParam, g0::RodriguesParam) = SVector(g0\g)

Base.angle(g::RodriguesParam) = angle(UnitQuaternion(g))


function rotmat(g::RodriguesParam)
    ghat = skew(SVector(g))
    I + 2*ghat*(ghat + I)/(1+norm2(g))
end

function ∇rotate(g0::RodriguesParam, r)
    g = SVector(g0)
    ghat = skew(g)
    n1 = 1/(1 + g'g)
    gxr = cross(g,r) + r
    d1 = ghat*gxr * -2*n1^2 * g'
    d2 = -(ghat*skew(r) + skew(gxr))*n1
    return 2d1 + 2d2
end

function ∇composition1(g2::RodriguesParam, g1::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)

    N = g2 + g1 + g2 × g1
    D = 1/(1 - g2'g1)
    (I + skew(g2) + D*N*g2')*D
end

function ∇²composition1(g2::RodriguesParam, g1::RodriguesParam, b::SVector{3})
    g2 = SVector(g2)
    g1 = SVector(g1)

    N = g2 + g1 + g2 × g1  # 3x1
    D = 1/(1 - g2'g1)  # scalar
    dN = I + skew(g2)
    dD = D^2*g2'
    return g2*b'*(N*(2*D*dD) + D^2*dN) + (I - skew(g2))*b*dD
end

function ∇composition2(g2::RodriguesParam, g1::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)

    N = g2 + g1 + g2 × g1
    D = 1/(1 - g2'g1)
    (I - skew(g1) + D*N*g1')*D
end

function ∇differential(g::RodriguesParam)
    g = SVector(g)
    (I + skew(g) + g*g')
end

function ∇²differential(g::RodriguesParam, b::SVector{3})
    g = SVector(g)
    return g*b'*(2g*g' + I + skew(g)) + (I - skew(g))*b*g'
end



############################################################################################
#                             Roll, Pitch, Yaw Euler Angles
############################################################################################

""" $(TYPEDEF)
Roll-pitch-yaw Euler angles.
"""
struct RPY{T} <: Rotation
    ϕ::T  # roll
    θ::T  # pitch
    ψ::T  # yaw
end

RPY(e::SVector{3,T}) where T = RPY{T}(e[1], e[2], e[3])
RPY(R::SMatrix{3,3,T}) where T =  RPY(rotmat_to_rpy(R))
RPY(q::UnitQuaternion) = RPY(rotmat(q))
RPY(p::MRP) = RPY(rotmat(p))
(::Type{<:RPY})(e::RPY) = e
(::Type{<:RPY})(::Type{T},x,y,z) where T = RPY{T}(T(x),T(y),T(z))
function RPY(ϕ::T1,θ::T2,ψ::T3) where {T1,T2,T3}
    T = promote_type(T1,T2)
    T = promote_type(T,T3)
    RPY(T(ϕ), T(θ), T(ψ))
end

SVector(e::RPY{T}) where T = SVector{3,T}(e.ϕ, e.θ, e.ψ)

Base.rand(::Type{RPY{T}}) where T = RPY(rand(UnitQuaternion{T}))
Base.rand(::Type{RPY}) = RPY(rand(UnitQuaternion))
Base.zero(::Type{RPY{T}}) where T = RPY(zero(T), zero(T), zero(T))
Base.zero(::Type{RPY}) = RPY(0.0, 0.0, 0.0)

roll(e::RPY) = e.ϕ
pitch(e::RPY) = e.θ
yaw(e::RPY) = e.ψ

Base.angle(e::RPY) = angle(UnitQuaternion(e))

(≈)(e1::RPY, e2::RPY) = rotmat(e1) ≈ rotmat(e2)

@inline rotmat(e::RPY) = rotmat(e.ϕ, e.θ, e.ψ)

function rotmat(ϕ, θ, ψ)
    # Equivalent to RotX(e[1])*RotY(e[2])*RotZ(e[3])
    sϕ,cϕ = sincos(ϕ)
    sθ,cθ = sincos(θ)
    sψ,cψ = sincos(ψ)
    # A = @SMatrix [
    #     cθ*cψ          -cθ*sψ              sθ;
    #     sϕ*sθ*cψ+cϕ*sψ -sϕ*sθ*sψ + cϕ*cψ  -cθ*sϕ;
    #    -cϕ*sθ*cψ+sϕ*sψ  cϕ*sθ*sψ + sϕ*cψ   cθ*cϕ
    # ]
    A = @SMatrix [
        cψ*cθ - sϕ*sψ*sθ   -cϕ*sψ  cψ*sθ + cθ*sϕ*sψ;
        cθ*sψ + cψ*sϕ*sθ    cϕ*cψ  sψ*sθ - cψ*cθ*sϕ;
        -cϕ*sθ              sϕ          cϕ*cθ;
    ]
end

(*)(e::RPY, r::SVector{3}) = rotmat(e)*r
(\)(e::RPY, r::SVector{3}) = rotmat(e)'r

function rotmat_to_rpy(R::SMatrix{3,3,T}) where T
    # ψ = atan(-R[1,2], R[1,1])
    # ϕ = atan(-R[2,3], R[3,3])
    # θ = asin(R[1,3])
    θ = atan(-R[3,1], R[3,3])
    ψ = atan(-R[1,2], R[2,2])
    ϕ = asin(R[3,2])
    return SVector{3,T}(ϕ, θ, ψ)
end

function from_rotmat(R::SMatrix{3,3,T}) where T
    ϕ,θ,ψ = rotmat_to_rpy(R)
    return RPY(ϕ, θ, ψ)
end

function (*)(e2::RPY, e1::RPY)
    from_rotmat(rotmat(e2)*rotmat(e1))
end

function (\)(e1::RPY, e2::RPY)
    from_rotmat(rotmat(e1)'rotmat(e2))
end

function (/)(e1::RPY, e2::RPY)
    from_rotmat(rotmat(e1)*rotmat(e2)')
end

function (⊖)(e1::RPY, e2::RPY)
    SVector(e1\e1)
end

function kinematics(e::RPY, ω::SVector{3})
    sθ,cθ = sincos(e.θ)
    sϕ,cϕ = sincos(e.ϕ)
    A = @SMatrix [
        cθ 0 -cϕ*sθ;
        0  1  sϕ;
        sθ 0  cϕ*cθ
    ]
    A\ω
end

function ∇rotate(e::RPY, r::SVector{3})
    rotate(e) = RPY(e)*r
    ForwardDiff.jacobian(rotate, SVector(e))
end

function ∇composition1(e2::RPY, e1::RPY)
    R2 = rotmat(e2)
    rotate(e) = rotmat_to_rpy(R2*rotmat(e[1],e[2],e[3]))
    ForwardDiff.jacobian(rotate,SVector(e1))
end

function ∇composition2(e2::RPY, e1::RPY)
    R1 = rotmat(e1)
    rotate(e) = rotmat_to_rpy(rotmat(e[1],e[2],e[3])*R1)
    ForwardDiff.jacobian(rotate,SVector(e2))
end

############################################################################################
#                                 CONVERSIONS
############################################################################################
""" Convert from a rotation matrix to a unit quaternion
Uses formula from Markely and Crassidis's book
    "Fundamentals of Spacecraft Attitude Determination and Control" (2014), section 2.9.3
"""
function rotmat_to_quat(A::SMatrix{3,3,T}) where T
    trA = tr(A)
    v,i = findmax(diag(A))
    if trA > v
        i = 1
    else
        i += 1
    end
    if i == 1
        q = UnitQuaternion(1+trA, A[2,3]-A[3,2], A[3,1]-A[1,3], A[1,2]-A[2,1])
    elseif i == 2
        q = UnitQuaternion(A[2,3]-A[3,2], 1 + 2A[1,1] - trA, A[1,2]+A[2,1], A[1,3]+A[3,1])
    elseif i == 3
        q = UnitQuaternion(A[3,1]-A[1,3], A[2,1]+A[1,2], 1+2A[2,2]-trA, A[2,3]+A[3,2])
    elseif i == 4
        q = UnitQuaternion(A[1,2]-A[2,1], A[3,1]+A[1,3], A[3,2]+A[2,3], 1 + 2A[3,3] - trA)
    end
    return normalize(inv(q))
end

UnitQuaternion(e::RPY) = rotmat_to_quat(rotmat(e))
(::Type{<:RPY})(q::UnitQuaternion) = from_rotmat(rotmat(q))

function (::Type{<:UnitQuaternion})(p::MRP)
    p = SVector(p)
    n2 = p'p
    M = 2/(1+n2)
    UnitQuaternion{MRPMap}((1-n2)/(1+n2), M*p[1], M*p[2], M*p[3])
end

(::Type{<:RodriguesParam})(p::MRP) = RodriguesParam(UnitQuaternion(p))

(::Type{<:RPY})(p::MRP) = RPY(UnitQuaternion(p))

############################################################################################
#                             INVERSE RETRACTION MAPS
############################################################################################
(::Type{ExponentialMap})(q::UnitQuaternion) = scaling(ExponentialMap)*logm(q)

(::Type{VectorPart})(q::UnitQuaternion) = scaling(VectorPart)*vector(q)

(::Type{CayleyMap})(q::UnitQuaternion) = scaling(CayleyMap) * vector(q)/q.s

(::Type{MRPMap})(q::UnitQuaternion) = scaling(MRPMap)*vector(q)/(1+q.s)

(::Type{IdentityMap})(q::UnitQuaternion) = SVector(q)

function jacobian(::Type{ExponentialMap}, q::UnitQuaternion, eps=1e-5)
    μ = scaling(ExponentialMap)
    s = scalar(q)
    v = vector(q)
    θ2 = v'v
    θ = sqrt(θ2)
    datan = 1/(θ2 + s^2)
    ds = -datan*v

    if θ < eps
        return 2*μ*[ds (v*v' + I)/s]
    else
        atanθ = atan(θ,s)
        dv = ((s*datan - atanθ/θ)v*v'/θ + atanθ*I )/θ
        d0 = ((s*datan - atanθ/θ)v*v'/θ^2 + atanθ/θ*I )
        d1 = (s*datan - atanθ/θ)
        d2 = v*v'/θ2
        d3 = atanθ/θ * I
        return 2*μ*[ds dv]
    end
end

function ∇jacobian(::Type{ExponentialMap}, q::UnitQuaternion, b::SVector{3}, eps=1e-5)
    μ = scaling(ExponentialMap)
    s = scalar(q)
    v = vector(q)
    θ2 = v'v
    θ = sqrt(θ2)
    datan = 1/(θ2 + s^2)
    ds = -datan*v

    if θ < eps
        # return 2*μ*[b'ds; (v*v'b + b)/s]
        return 2*μ*[b'*(datan^2*2s*v) -b'datan*I;
                    -(v*v'b +b)/s^2 (I*(v'b) + v*b')/s]
    else
        dsds = 2b's*datan^2*v
        dsdv = b'*(-datan*I + 2datan^2*v*v')

        atanθ = atan(θ,s)
        d1 = (s*datan - atanθ/θ)
        d2 = v*v'b/θ2
        d3 = atanθ/θ*b
        d1ds = (datan - 2s^2*datan^2 + datan)
        dvds = d1ds*d2 - datan*b

        d1dv =  (-2s*datan^2*v' - s*datan*v'/θ^2 + atanθ/θ^3*v')
        d2dv = (I*(v'b) + v*b')/θ2 - 2(v*v'b)/θ^4 * v'
        d3dv = b*(s*datan*v'/θ^2 - atanθ/θ^3*v')
        dvdv = d2*d1dv + d1*d2dv + d3dv

        # return 2*μ*[ds'b; dv'b]
        return 2*μ*@SMatrix [
            dsds    dsdv[1] dsdv[2] dsdv[3];
            dvds[1] dvdv[1] dvdv[4] dvdv[7];
            dvds[2] dvdv[2] dvdv[5] dvdv[8];
            dvds[3] dvdv[3] dvdv[6] dvdv[9];
        ]
    end
end

function jacobian(::Type{VectorPart}, q::UnitQuaternion)
    μ = scaling(VectorPart)
    return @SMatrix [0. μ 0 0;
                     0. 0 μ 0;
                     0. 0 0 μ]
end

function ∇jacobian(::Type{VectorPart}, q::UnitQuaternion, b::SVector{3})
    μ = scaling(VectorPart)
    @SMatrix zeros(4,4)
end

function jacobian(::Type{CayleyMap}, q::UnitQuaternion)
    μ = scaling(CayleyMap)
    si = 1/q.s
    return μ*@SMatrix [-si^2*q.x si 0 0;
                       -si^2*q.y 0 si 0;
                       -si^2*q.z 0 0 si]
end

function ∇jacobian(::Type{CayleyMap}, q::UnitQuaternion, b::SVector{3})
    μ = scaling(CayleyMap)
    si = 1/q.s
    v = vector(q)
    μ*@SMatrix [
        2*si^3*(v'b) -si^2*b[1] -si^2*b[2] -si^2*b[3];
       -si^2*b[1] 0 0 0;
       -si^2*b[2] 0 0 0;
       -si^2*b[3] 0 0 0;
    ]
end

function jacobian(::Type{MRPMap}, q::UnitQuaternion)
    μ = scaling(MRPMap)
    si = 1/(1+q.s)
    return μ*@SMatrix [-si^2*q.x si 0 0;
                       -si^2*q.y 0 si 0;
                       -si^2*q.z 0 0 si]
end

"""
Jacobian of G(q)'b, where G(q) = jacobian(MRPMap,q)
"""
function ∇jacobian(::Type{MRPMap}, q::UnitQuaternion, b::SVector{3})
    μ = scaling(MRPMap)
    si = 1/(1+q.s)
    v = vector(q)
    μ * @SMatrix [
        2*si^3*(v'b) -si^2*b[1] -si^2*b[2] -si^2*b[3];
       -si^2*b[1] 0 0 0;
       -si^2*b[2] 0 0 0;
       -si^2*b[3] 0 0 0;
    ]
end

jacobian(::Type{IdentityMap}, q::UnitQuaternion) = I

inverse_map_jacobian(q::R) where R<:Rotation = I
inverse_map_jacobian(q::UnitQuaternion{T,D}) where {T,D} = jacobian(D,q)

inverse_map_∇jacobian(q::R, b::SVector{3}) where R<:Rotation = I*0
inverse_map_∇jacobian(q::UnitQuaternion{T,D}, b::SVector{3}) where {T,D} = ∇jacobian(D, q, b)
