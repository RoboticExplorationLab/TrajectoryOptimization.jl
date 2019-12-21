using Quaternions
using Test
q1 = nquatrand()
q2 = nquatrand()

dq = inv(q1)*q2
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
