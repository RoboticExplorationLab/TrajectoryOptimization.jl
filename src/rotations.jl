import Base: +, -, *, /, exp, log, ≈, inv, conj
import LinearAlgebra: norm2

export
    Rotation,
    UnitQuaternion,
    MRP,
    RPY,
    ExponentialMap,
    VectorPart,
    ModifiedRodriguesParam

export
    differential_rotation,
    scalar,
    vector,
    logm,
    expm,
    kinematics,
    rotmat,
    Lmult,
    Rmult,
    Vmat,
    Tmat,
    ⊕,
    ⊖


function skew(v::AbstractVector)
    @assert length(v) == 3
    @SMatrix [0   -v[3]  v[2];
              v[3] 0    -v[1];
             -v[2] v[1]  0]
end

abstract type Rotation end

abstract type DifferentialRotation end
abstract type VectorPart <: DifferentialRotation end
abstract type ExponentialMap <: DifferentialRotation end
abstract type ModifiedRodriguesParam <: DifferentialRotation end

const DEFAULT_QUATDIFF = VectorPart

""" $(TYPEDEF)
4-parameter attitute representation that is singularity-free. Quaternions with unit norm
represent a double-cover of SO(3). The `UnitQuaternion` does NOT strictly enforce the unit
norm constraint, but certain methods will assume you have a unit quaternion. The
`UnitQuaternion` type is parameterized by the linearization method, which maps quaternions
to the 3D plane tangent to the 4D unit sphere. Follows the Hamilton convention for quaternions.

There are currently 3 methods supported:
* `VectorPart` - uses the vector (or imaginary) part of the quaternion
* `ExponentialMap` - the most common approach, uses the exponential and logarithmic maps
* `ModifiedRodriguesParam` - or Modified Rodrigues Parameter, is a sterographic projection of the 4D unit sphere
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
UnitQuaternion{D}(s::T,x::T,y::T,z::T) where {T,D} = UnitQuaternion{T,D}(s,x,y,z)

UnitQuaternion{D}(q::SVector{4}) where D = UnitQuaternion{D}(q[1],q[2],q[3],q[4])
UnitQuaternion{D}(r::SVector{3}) where D = UnitQuaternion{D}(0.0, r[1],r[2],r[3])
UnitQuaternion{D}(q::UnitQuaternion) where D = UnitQuaternion{D}(q.s, q.x, q.y, q.z)

UnitQuaternion(r::SVector{3}) where D = UnitQuaternion{DEFAULT_QUATDIFF}(0.0, r[1],r[2],r[3])

Base.rand(::Type{<:UnitQuaternion{T}}) where T =
    normalize(UnitQuaternion(randn(T), randn(T), randn(T), randn(T)))

SVector(q::UnitQuaternion{T}) where T = SVector{4,T}(q.s, q.x, q.y, q.z)

scalar(q::UnitQuaternion) = q.s
vector(q::UnitQuaternion{T}) where T = SVector{3,T}(q.x, q.y, q.z)
vecnorm(q::UnitQuaternion) = sqrt(q.x^2 + q.y^2 + q.z^2)

conj(q::UnitQuaternion) = UnitQuaternion(q.s, -q.x, -q.y, -q.z)
inv(q::UnitQuaternion) = conj(q)

LinearAlgebra.norm2(q::UnitQuaternion) = q.s^2 + q.x^2 + q.y^2 + q.z^2
LinearAlgebra.norm(q::UnitQuaternion) = sqrt(q.s^2 + q.x^2 + q.y^2 + q.z^2)
LinearAlgebra.I(::Type{UnitQuaternion}) = UnitQuaternion(1.0, 0.0, 0.0, 0.0)

(≈)(q::UnitQuaternion, u::UnitQuaternion) = q.s ≈ u.s && q.x ≈ u.x && q.y ≈ u.y && q.z ≈ u.z

function LinearAlgebra.normalize(q::UnitQuaternion{T,D}) where {T,D}
    n = norm(q)
    UnitQuaternion{T,D}(q.s/n, q.x/n, q.y/n, q.z/n)
end

function (*)(q::UnitQuaternion{T1,D}, w::UnitQuaternion{T2,D}) where {T1,T2,D}
    T = promote_type(T1, T2)
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
    M = sθ/θ
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

function (⊕)(q::UnitQuaternion{T,ExponentialMap}, δq::SVector{3}) where T
    q*expm(δq)
end

function (⊕)(q::UnitQuaternion{T,VectorPart}, δq::SVector{3}) where T
    q*UnitQuaternion{VectorPart}(sqrt(1 - δq[1]^2 - δq[2]^2 - δq[3]^2),
        δq[1], δq[2], δq[3])
end

function (⊖)(q::UnitQuaternion{T,ExponentialMap}, q0::UnitQuaternion) where T
    logm(inv(q0)*q)
end

function (⊖)(q::UnitQuaternion{T,VectorPart}, q0::UnitQuaternion) where T
    vector(inv(q0)*q)
end

"""
Jacobian of q ⊕ ϕ. Useful for converting Jacobians from R⁴ to R³ and
    correctly account for unit norm constraint. Jacobians for different
    differential quaternion parameterization are the same up to a constant.
"""
function ∇differential(q::UnitQuaternion{T,D}) where {T,D}
    ∇differential(D) * @SMatrix [
        -q.x -q.y -q.z;
         q.s -q.z  q.y;
         q.z  q.s -q.x;
        -q.y  q.x  q.s;
    ]
end

@inline ∇differential(::Type{ExponentialMap}) = 2.0
@inline ∇differential(::Type{VectorPart}) = 1.0
@inline ∇differential(::Type{ModifiedRodriguesParam}) = 0.5

"Lmult(q2)q1 returns a vector equivalent to q2*q1 (quaternion multiplication)"
function Lmult(q::UnitQuaternion)
    @SMatrix [
        q.s -q.x -q.y -q.z;
        q.x  q.s -q.z  q.y;
        q.y  q.z  q.s -q.x;
        q.z -q.y  q.x  q.s;
    ]
end

"Rmult(q1)q2 return a vector equivalent to q2*q1 (quaternion multiplication)"
function Rmult(q::UnitQuaternion)
    @SMatrix [
        q.s -q.x -q.y -q.z;
        q.x  q.s  q.z -q.y;
        q.y -q.z  q.s  q.x;
        q.z  q.y -q.x  q.s;
    ]
end

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
function MRP(q::UnitQuaternion)
    M = 1/(1+q.s)
    MRP(q.x*M, q.y*M, q.z*M)
end

function UnitQuaternion(p::MRP)
    n = norm2(p)
    s = (1-n)/(1+n)
    M = 2/(1+n)
    UnitQuaternion{ModifiedRodriguesParam}(s, p.x*M, p.y*M, p.z*M)
end

function rotmat(p::MRP)
    p = SVector(p)
    P = skew(p)
    R2 = I + 4( (1-p'p)I + 2*P )*P/(1+p'p)^2
end

SVector(p::MRP{T}) where T = SVector{3,T}(p.x, p.y, p.z)

Base.rand(::Type{MRP{T}}) where T = MRP(rand(UnitQuaternion{T}))

LinearAlgebra.norm(p::MRP) = sqrt(p.x^2 + p.y^2 + p.z^2)
LinearAlgebra.norm2(p::MRP) = p.x^2 + p.y^2 + p.z^2

function (*)(p2::MRP, p1::MRP)
    p2, p1 = SVector(p2), SVector(p1)
    MRP(((1-p2'p2)*p1 + (1-p1'p1)*p2 - cross(2p1, p2) ) / (1+p1'p1*p2'p2 - 2p1'p2))
end

(*)(p::MRP, r::SVector) = UnitQuaternion(p)*r

function kinematics(p::MRP, ω)
    p = SVector(p)
    A = @SMatrix [
        1 + p[1]^2 - p[2]^2 - p[3]^2  2(p[1]*p[2] - p[3])      2(p[1]*p[3] + p[2]);
        2(p[2]*p[1] + p[3])            1-p[1]^2+p[2]^2-p[3]^2   2(p[2]*p[3] - p[1]);
        2(p[3]*p[1] - p[2])            2(p[3]*p[2] + p[1])      1-p[1]^2-p[2]^2+p[3]^2]
    0.25*A*ω
end

(⊕)(p::MRP, δp::SVector{3}) = MRP(p.x+δp[1], p.y+δp[2], p.z+δp[3])
(⊖)(p::MRP, p0::SVector{3}) = MRP(p.x-p0.x, p.y-p0.y, p.z-p0.z)

function ∇rotate(p::MRP, r)
    p = SVector(p)
    4( (1-p'p)*skew(r)*(4p*p'/(1+p'p) - I) - (4/(1+p'p)*skew(p) + I)*2*skew(p)*r*p'
      - 2*(skew(p)*skew(r) + skew(skew(p)*r)))/(1+p'p)^2
end

function ∇composition2(p2::MRP, p1::MRP)
    # this is slower than ForwardDiff
    n1 = norm(p1)
    n2 = norm(p2)
    p1, p2 = SVector(p1), SVector(p2)
    N = (1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2)
    D = 1+n1*n2 - 2p1'p2
    (((1-n1)*I - 2p1*p2' - skew(2p1))*D - N*(2n1*p2' - 2p1'))/D^2
end

function ∇composition1(p2::MRP, p1::MRP)
    # this is slower than ForwardDiff
    n1 = norm(p1)
    n2 = norm(p2)
    p1, p2 = SVector(p1), SVector(p2)
    N = (1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2)
    D = 1+n1*n2 - 2p1'p2
    (((1-n2)*I - 2p2*p1' + skew(2p2))*D - N*(2n2*p1' - 2p2'))/D^2
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

SVector(e::RPY{T}) where T = SVector{3,T}(e.ϕ, e.θ, e.ψ)

Base.rand(::Type{RPY{T}}) where T = RPY(rand(UnitQuaternion{T}))
Base.rand(::Type{RPY}) = RPY(rand(UnitQuaternion))

roll(e::RPY) = e.ϕ
pitch(e::RPY) = e.θ
yaw(e::RPY) = e.ψ

(≈)(e1::RPY, e2::RPY) = rotmat(e1) ≈ rotmat(e2)

@inline rotmat(e::RPY) = rotmat(e.ϕ, e.θ, e.ψ)

function rotmat(ϕ, θ, ψ)
    # Equivalent to RotX(e[1])*RotY(e[2])*RotZ(e[3])
    sϕ,cϕ = sincos(ϕ)
    sθ,cθ = sincos(θ)
    sψ,cψ = sincos(ψ)
    A = @SMatrix [
        cθ*cψ          -cθ*sψ              sθ;
        sϕ*sθ*cψ+cϕ*sψ -sϕ*sθ*sψ + cϕ*cψ  -cθ*sϕ;
       -cϕ*sθ*cψ+sϕ*sψ  cϕ*sθ*sψ + sϕ*cψ   cθ*cϕ
    ]
end

(*)(e::RPY, r::SVector{3}) = rotmat(e)*r

function rotmat_to_rpy(R::SMatrix{3,3,T}) where T
    ψ = atan(-R[1,2], R[1,1])
    ϕ = atan(-R[2,3], R[3,3])
    θ = asin(R[1,3])
    return SVector{3,T}(ϕ, θ, ψ)
end

function from_rotmat(R::SMatrix{3,3,T}) where T
    ϕ,θ,ψ = rotmat_to_rpy(R)
    return RPY(ϕ, θ, ψ)
end

function (*)(e2::RPY, e1::RPY)
    from_rotmat(rotmat(e2)*rotmat(e1))
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



# Differential Rotations
function differential_rotation(δq::UnitQuaternion{T,VectorPart}) where T
    SVector{3}(δq.x, δq.y, δq.z)
end

function differential_rotation(δq::UnitQuaternion{T,ExponentialMap}) where T
    logm(δq)
end

function differential_rotation(δq::UnitQuaternion{T,ModifiedRodriguesParam}) where T
    # vector(δq)/(1+real(δq))
    s = 1+scalar(δq)
    SVector{3}(δq.x/s, δq.y/s, δq.z/s)
end
