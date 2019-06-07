using Rotations
using CoordinateTransformations
using StaticArrays
using LinearAlgebra

struct Quaternion{T}
    w::T
    x::T
    y::T
    z::T
end

Quaternion(v::AbstractVector) = Quaternion(v[1],v[2],v[3],v[4])
Quaternion(w::Real, v::AbstractVector) = Quaternion(w,v[1],v[2],v[3])
Quaternion(q::Quat) = Quaternion(q.w, q.x, q.y, q.z)
Quaternion(r::Rotation{3,T}) where T = Quaternion(Quat(r))
CoordinateTransformations.Quat(q::Quaternion) = Quat(SVector(q)...)
Base.vec(q::Quaternion) = SVector(q.x, q.y, q.z)
StaticArrays.SVector(q::Quaternion) = SVector(q.w, q.x, q.y, q.z)
scalar(q::Quaternion) = q.w


function Base.:*(q2::Quaternion, q1::Quaternion)
    w = scalar(q1)*scalar(q2) - vec(q1)'vec(q2)
    v = scalar(q1)*vec(q2) + scalar(q2)*vec(q1) + vec(q2) × vec(q1)
    Quaternion(w,v)
end

Base.:*(A::AbstractMatrix, q::Quaternion) = A*SVector(q)
Base.:*(q::Quaternion, A::AbstractMatrix) = SVector(q)'A
Base.:*(a::Real, q::Quaternion) = Quaternion(a*scalar(q), a*vec(q))
function Base.:*(q::Quaternion, r::AbstractVector)
    @assert length(r) == 3
    # conj(q)*r
    vec(q*Quaternion(zero(eltype(r)),r)*inv(q))
    # s,v = scalar(q), vec(q)
    # [0; v*v'r + s^2*r + 2s*skew(v)*r + skew(v)^2*r]
end

Base.inv(q::Quaternion) = Quaternion(q.w, -q.x, -q.y, -q.z)
LinearAlgebra.normalize(q::Quaternion) = Quaternion(normalize(SVector(q)))

function skew(v::AbstractVector)
    @assert length(v) == 3
    @SMatrix [0   -v[3]  v[2];
              v[3] 0    -v[1];
             -v[2] v[1]  0]
end

function Lmult(q::Quaternion)
    @SMatrix [q.w -q.x -q.y -q.z
              q.x  q.w -q.z  q.y;
              q.y  q.z  q.w -q.x;
              q.z -q.y  q.x  q.w];
end

function Rmult(q::Quaternion)
    @SMatrix [q.w -q.x -q.y -q.z
              q.x  q.w  q.z -q.y;
              q.y -q.z  q.w  q.x;
              q.z  q.y -q.x  q.w];
end

function Base.conj(q::Quaternion)
    s,v = scalar(q), vec(q)
    vhat = skew(v)
    # C = [s^2 + v'v v'vhat; vhat*v v*v' + s^2*I + 2s*vhat + vhat*vhat]
    C = v*v' + s^2*I + 2s*vhat + vhat*vhat
    return C
end

function deriv_conj(q::Quaternion, r::AbstractVector)
    s,v = scalar(q), vec(q)
    j2 = 2(s*I + skew(v))*r
    j3 = v*r' + I*v'r - 2s*skew(r) - skew(v)*skew(r) - skew(skew(v)*r)
    return [j2 j3]
end
