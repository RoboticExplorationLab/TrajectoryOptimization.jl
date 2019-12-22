using BenchmarkTools
using Quaternions
using Rotations
using Test
import Base: ==, ≈
Rotations.Quat(q::Quaternion) = Rotations.Quat(q.s, q.v1, q.v2, q.v3)
UnitQuaternion(q::Quaternion) = UnitQuaternion(q.s, q.v1, q.v2, q.v3)
SVector(q::Quaternion) = SVector{4}(q.s, q.v1, q.v2, q.v3)

(==)(q::Quaternion, u::UnitQuaternion) = q.s == u.s && q.v1 == u.x && q.v2 == u.y && q.v3 == u.z
(==)(u::UnitQuaternion, q::Quaternion) = q == u
(==)(q::Quat, u::UnitQuaternion) = q.w == u.s && q.x == u.x && q.y == u.y && q.z == u.z
(==)(u::UnitQuaternion, q::Quat) = q == u
(==)(q::Quaternion, u::Quat) = q.s == u.w && q.v1 == u.x && q.v2 == u.y && q.v3 == u.z
(==)(u::Quat, q::Quaternion) = q == u

(≈)(q::Quaternion, u::UnitQuaternion) = q.s ≈ u.s && q.v1 ≈ u.x && q.v2 ≈ u.y && q.v3 ≈ u.z
(≈)(u::UnitQuaternion, q::Quaternion) = q ≈ u
(≈)(q::Quat, u::UnitQuaternion) = q.w ≈ u.s && q.x ≈ u.x && q.y ≈ u.y && q.z ≈ u.z
(≈)(u::UnitQuaternion, q::Quat) = q ≈ u
(≈)(q::Quaternion, u::Quat) = q.s ≈ u.w && q.v1 ≈ u.x && q.v2 ≈ u.y && q.v3 ≈ u.z
(≈)(u::Quat, q::Quaternion) = q ≈ u

q1 = nquatrand()
q2 = nquatrand()

r1 = Rotations.Quat(q1)
r2 = Rotations.Quat(q2)

u1 = UnitQuaternion(q1)
u2 = UnitQuaternion(q2)

q1 == u1
u1 == q1
r1 == u1
u1 == r1
q1 == r1
r1 == q1

q1*q2 ≈ r1*r2
q1*q2 ≈ u1*u2
@btime $q1*$q2
@btime $r1*$r2
@btime $u1*$u2


r = @SVector rand(3)
q1*r ≈ r1*r
q1*r ≈ u1*r
@btime $q1*$r
@btime $r1*$r
@btime $u1*$r

exp(q1) ≈ exp(u1)
@btime exp($q1)
@btime exp($u1)

log(q1) ≈ log(u1)
@btime log($q1)
@btime log($u1)

ω = @SVector rand(3)
kinematics(q::Quaternion, ω) = SVector(0.5*q*Quaternion(0, ω[1], ω[2], ω[3]))
kinematics(u1, ω) ≈ kinematics(q1, ω)
kinematics(u1, ω)
@btime kinematics($q1, $ω)
@btime kinematics($u1, $ω)

ϕ = 0.1*@SVector [1,0,0]
@test logm(expm(ϕ)) ≈ ϕ
@test expm(logm(u1)) ≈ u1
@btime expm($ϕ)
@btime logm($u1)
@test norm(u1 ⊕ ϕ) == 1
@test u1 ⊖ u2 isa SVector{3}
@test (u1 ⊕ ϕ) ⊖ u1 ≈ ϕ


q = u1
rhat = UnitQuaternion(r)
@test q*r ≈ Vmat()*Lmult(q)*Rmult(q)'*Vmat()'r
@test q*r ≈ Vmat()*Lmult(q)*Lmult(rhat)*Tmat()*SVector(q)
@test q*r ≈ Vmat()*Rmult(q)'*Rmult(rhat)*SVector(q)

@test ForwardDiff.jacobian(q->UnitQuaternion(q)*r,SVector(q)) ≈ ∇rotate(q,r)
@btime ForwardDiff.jacobian(q->UnitQuaternion(q)*$r,SVector($q))
@btime ∇rotate($q,$r)

ForwardDiff.jacobian(q->SVector(u2*UnitQuaternion(q)),SVector(u1)) ≈ ∇composition1(u2,u1)
ForwardDiff.jacobian(q->SVector(UnitQuaternion(q)*u1),SVector(u2)) ≈ ∇composition2(u2,u1)
@btime ForwardDiff.jacobian(q->SVector($u2*UnitQuaternion(q)),SVector($u1))
@btime ∇composition1($u2,$u1)
@btime ForwardDiff.jacobian(q->SVector(UnitQuaternion(q)*$u1),SVector($u2))
@btime ∇composition2($u2,$u1)


# RPY
e0 = @SVector [deg2rad(45), deg2rad(60), deg2rad(20)]
e1 = RPY(e0...)
e2 = RotXYZ(e0...)

@test rotmat(e1) == RotX(e0[1])*RotY(e0[2])*RotZ(e0[3])
@test rotmat(e1) ≈ e2

@test e1*r ≈ e2*r
@btime $e2*$r
@btime $e1*$r

R = rotmat(e1)
e1_ = from_rotmat(rotmat(e1))
@test e1_.θ ≈ e1.θ
@test e1_.ψ ≈ e1.ψ
@test e1_.ϕ ≈ e1.ϕ

@test rotmat(e1*e1) ≈ RotXYZ(e2*e2)

e1 = RPY(rand(3)...)
e2 = RPY(rand(3)...)
R = rotmat(e2*e1)

sϕ1,cϕ1 = sincos(e1.ϕ)
sθ1,cθ1 = sincos(e1.θ)
sψ1,cψ1 = sincos(e1.ψ)

sϕ2,cϕ2 = sincos(e2.ϕ)
sθ2,cθ2 = sincos(e2.θ)
sψ2,cψ2 = sincos(e2.ψ)

R11 = cθ2*cψ2*(cθ1*cψ1) - cθ2*sψ2*(sϕ1*sθ1*cψ1+cϕ1*sψ1) + sθ2*(-cϕ1*sθ1*cψ1+sϕ1*sψ1)
R11 ≈ R[1,1]
R12 = cθ2*cψ2*(-cθ1*sψ1) - cθ2*sψ2*(-sϕ1*sθ1*sψ1+cϕ1*cψ1) + sθ2*(cϕ1*sθ1*sψ1+sϕ1*cψ1)
R12 ≈ R[1,2]

dxϕ1 = -cθ2*sψ2*(cϕ1*sθ1*cψ1-sϕ1*sψ1) + sθ2*( sϕ1*sθ1*cψ1+cϕ1*sψ1)
dxθ1 = -cθ2*cψ2*(sθ1*cψ1) - cθ2*sψ2*(sϕ1*cθ1*cψ1) + sθ2*(-cϕ1*cθ1*cψ1)
dxψ1 = -cθ2*cψ2*(cθ1*sψ1) - cθ2*sψ2*(-sϕ1*sθ1*sψ1+cϕ1*cψ1) + sθ2*( cϕ1*sθ1*sψ1+sϕ1*cψ1)

dyϕ1 = -cθ2*sψ2*(-sϕ1*sθ1*sψ1+cϕ1*cψ1) + sθ2*(cϕ1*sθ1*sψ1+sϕ1*cψ1)

dψdϕ1 = -R11/(R11^2 + R12^2)*dϕ1 + (R12)/(R11^2 + R12^2)*dϕ1

vals1 = @SVector [e1.ϕ, e1.θ, e1.ψ]
R2 = rotmat(e2)
f(e) = rotmat_to_rpy(R2*rotmat(e...))
f(vals1)
ForwardDiff.jacobian(f, vals1)

@btime ∇rotation1($e2, $e1)
@btime ∇rotation2($e2, $e1)

# MRPs
p1 = MRP(rand(3)...)
p2 = MRP(rand(3)...)
@btime $p2*$p1
@btime $p2*$r
q = UnitQuaternion(p2)



dq = inv(q1)*q2k
@test differential_rotation(VectorPart,dq) ≈ vector(dq)
@test differential_rotation(ExponentialMap,dq) ≈ logm(dq)
@test logm(dq) ≈ 2*vector(log(dq))
@test differential_rotation(ModifiedRodriguesParam,dq) ≈ vector(dq)/(1+real(dq))

function diff_allocs(dq)
    allocs  = @allocated differential_rotation(VectorPart,dq)
    allocs += @allocated differential_rotation(ExponentialMap,dq)
    allocs += @allocated differential_rotation(ModifiedRodriguesParam,dq)
end
@test diff_allocs(dq) == 0

r = @SVector rand(3)
R = rotationmatrix(q1)
@test R*r ≈ q1*r

p = differential_rotation(ModifiedRodriguesParam,q1)
R1 = MRP_to_DCM(p)
@test R1*r ≈ R*r
@test MRP_rotate_vec(p,r) ≈ R*r

@test MRP_to_quat(p) ≈ q1
@test MRP_rotate_vec2(p,r) ≈ R*r



# Quadrotors
quad1 = Dynamics.Quadrotor()
quad2 = Dynamics.Quadrotor2{Quat{VectorPart}}()
x,u = rand(quad2)
dynamics(quad1,x,u) ≈ dynamics(quad2,x,u)
@btime dynamics($quad1,$x,$u)
@btime dynamics($quad2,$x,$u)

@btime state_diff($quad1,$x,$(1.1x))
@btime state_diff($quad2,$x,$(1.1x))
@btime state_diff_jacobian($quad1, $x)
@btime state_diff_jacobian($quad2, $x)

a = @SVector rand(3)
b = @SVector rand(3)
@test skew(a)*b == cross(a,b)
skew(b)*a == -skew(a)*b

p = a
r = b
f(x) = MRP_to_DCM(x)*r
@test ForwardDiff.jacobian(f,p) ≈ MRP_rotate_jacobian(p,r)

# Test MRP compositio
p1 = @SVector rand(3)
p2 = @SVector rand(3)
r = @SVector rand(3)
R1 = MRP_to_DCM(p1)
R2 = MRP_to_DCM(p2)
R3 = R2*R1
p3 = MRP_composition(p2,p1)
@test MRP_rotate_vec(p1,r) ≈ R1*r
@test MRP_rotate_vec(p2,r) ≈ R2*r
@test MRP_rotate_vec(p3,r) ≈ R3*r

ForwardDiff.jacobian(x->MRP_composition(x,p1), p2) ≈ MRP_composition_jacobian_p2(p2,p1)
ForwardDiff.jacobian(x->MRP_composition(p2,x), p1) ≈ MRP_composition_jacobian_p1(p2,p1)

using Rotations
RPY(e) = RotX(e[1])*RotY(e[2])*RotZ(e[3])
RPY(e) .≈ RPY_to_DCM(e)
