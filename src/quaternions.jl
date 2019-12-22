
@inline SVector(q::Quaternion) = SVector{4}(q.s, q.v1, q.v2, q.v3)
@inline vector(q::Quaternion) = SVector{3}(q.v1, q.v2, q.v3)

function logm(q::Quaternion)
    if q.norm
        M =  sqrt(q.v1^2 + q.v2^2 + q.v3^2)
        M = 2*atan(M, q.s) / M
        return SVector{3}(q.v1*M, q.v2*M, q.v3*M)
    else
        q = log(q)
        SVector{3}(2*q.v1, 2*q.v2, 2*q.v3)
    end
end

function Base.:*(q::Quaternion{Tq}, r::SVector{3}) where Tq
    qo = (-q.v1 * r[1] - q.v2 * r[2] - q.v3 * r[3],
           q.s  * r[1] + q.v2 * r[3] - q.v3 * r[2],
           q.s  * r[2] - q.v1 * r[3] + q.v3 * r[1],
           q.s  * r[3] + q.v1 * r[2] - q.v2 * r[1])

   T = promote_type(Tq, eltype(r))

   return similar_type(r, T)(-qo[1] * q.v1 + qo[2] * q.s  - qo[3] * q.v3 + qo[4] * q.v2,
                             -qo[1] * q.v2 + qo[2] * q.v3 + qo[3] * q.s  - qo[4] * q.v1,
                             -qo[1] * q.v3 - qo[2] * q.v2 + qo[3] * q.v1 + qo[4] * q.s)
end

function differential_rotation(::Type{VectorPart}, δq::Quaternion)
    SVector{3}(δq.v1, δq.v2, δq.v3)
end

function differential_rotation(::Type{ExponentialMap}, δq::Quaternion)
    logm(δq)
end

function differential_rotation(::Type{ModifiedRodriguesParam}, δq::Quaternion)
    # vector(δq)/(1+real(δq))
    s = 1+real(δq)
    SVector{3}(δq.v1/s, δq.v2/s, δq.v3/s)
end

function quat_diff_jacobian(q::Quaternion)
    w = real(q)
    x,y,z = vector(q)
    w,x,y,z = q.s, q.v1, q.v2, q.v3
    x,y,z = -x,-y,-z  # invert q
    @SMatrix [x  w -z  y;
              y  z  w -x;
              z -y  x  w];
end


function MRP_kinematics(p::SVector{3,T}) where T
    @SMatrix [
        1 + p[1]^2 - p[2]^2 - p[3]^2  2(p[1]*p[2] - p[3])      2(p[1]*p[3] + p[2]);
        2(p[2]*p[1] + p[3])            1-p[1]^2+p[2]^2-p[3]^2   2(p[2]*p[3] - p[1]);
        2(p[3]*p[1] - p[2])            2(p[3]*p[2] + p[1])      1-p[1]^2-p[2]^2+p[3]^2]
end


function MRP_to_DCM(p::SVector{3,T}) where T
    P = skew(p)
    R2 = I + 4( (1-p'p)I + 2*P )*P/(1+p'p)^2
end

"""
Rotate a vector r with MRP p.
Faster to convert to quaternion than to convert to rotmat
"""
function MRP_rotate_vec(p::SVector{3,T}, r::SVector{3,T}) where T
    MRP_to_quat(p)*r
end

function MRP_to_quat(p::SVector{3,T}) where T
    n = p'p
    s = (1-n)/(1+n)
    v = 2p/(1+n)
    Quaternion(s, v[1], v[2], v[3])
end


function MRP_rotate_jacobian(p,r)
    4( (1-p'p)*skew(r)*(4p*p'/(1+p'p) - I) - (4/(1+p'p)*skew(p) + I)*2*skew(p)*r*p'
      - 2*(skew(p)*skew(r) + skew(skew(p)*r)))/(1+p'p)^2
end

function MRP_composition(p2,p1)
    ((1-p2'p2)*p1 + (1-p1'p1)*p2 - cross(2p1, p2) ) / (1+p1'p1*p2'p2 - 2p1'p2)
end


function MRP_composition_jacobian_p2(p2,p1)
    # this is slower than ForwardDiff
    n1 = p1'p1
    n2 = p2'p2
    N = (1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2)
    D = 1+n1*n2 - 2p1'p2
    (((1-n1)*I - 2p1*p2' - skew(2p1))*D - N*(2n1*p2' - 2p1'))/D^2
end

function MRP_composition_jacobian_p1(p2,p1)
    # this is slower than ForwardDiff
    n1 = p1'p1
    n2 = p2'p2
    N = (1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2)
    D = 1+n1*n2 - 2p1'p2
    (((1-n2)*I - 2p2*p1' + skew(2p2))*D - N*(2n2*p1' - 2p2'))/D^2
end

function RPY_to_DCM(e)
    # Equivalent to RotX(e[1])*RotY(e[2])*RotZ(e[3])
    sϕ,cϕ = sin(e[1]), cos(e[1])
    sθ,cθ = sin(e[2]), cos(e[2])
    sψ,cψ = sin(e[3]), cos(e[3])
    A = @SMatrix [
        cθ*cψ          -cθ*sψ              sθ;
        sϕ*sθ*cψ+cϕ*sψ -sϕ*sθ*sψ + cϕ*cψ  -cθ*sϕ;
       -cϕ*sθ*cψ+sϕ*sψ  cϕ*sθ*sψ + sϕ*cψ   cθ*cϕ
    ]
end
