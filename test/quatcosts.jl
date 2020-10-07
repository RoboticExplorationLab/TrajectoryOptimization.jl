using Test
using TrajectoryOptimization
using RobotDynamics
using RobotDynamics: LieState
import RobotZoo.Quadrotor
using BenchmarkTools
using Rotations
using StaticArrays, LinearAlgebra, ForwardDiff
const TO = TrajectoryOptimization
const RD = RobotDynamics

import TrajectoryOptimization: DiagonalLieCost 



## Test constructors
P = (3,6)
Qs = [rand(p) for p in P]
s = LieState(UnitQuaternion,3,6)
R = rand(4)
dcost = DiagonalLieCost(s, Qs, R)
@test dcost.w == [1]
@test dcost.Q == [Qs[1]; Qs[2]]
@test length(dcost.Q) == sum(P)

# Pass in Diagonal R
dcost = DiagonalLieCost(s, Qs, Diagonal(R))
@test dcost.R.diag == R

# Pass in Diagonal Q for vector parts
Q = Diagonal(vcat(Qs...))
q = rand(9)
dcost = DiagonalLieCost(s, Q, Diagonal(R), q=q, w=[5])
@test dcost.Q == [Qs[1]; Qs[2]]
@test dcost.q == q 
@test dcost.w[1] === 5.0

# Pass in Diagonal Q for all states
Q = Diagonal(rand(13))
q = rand(13)
dcost = DiagonalLieCost(s, Q, Diagonal(R), q=q)
@test dcost.Q == [Q[1:3]; Q[8:13]]
@test dcost.q == [q[1:3]; q[8:13]]
@test dcost.w[1] ≈ sum(Q[4:7])

# Pass in weight, overriding default behavior of summing diagonal elements
dcost = DiagonalLieCost(s, Q, Diagonal(R), q=q, w=[6])
@test dcost.w[1] === 6.0


## Set up
model = Quadrotor{UnitQuaternion}()
x, u = rand(model)
s = LieState(model)
Q = rand(RobotDynamics.state_dim_vec(s))
R = rand(control_dim(model))
q = rand(RobotDynamics.state_dim_vec(s))
r = rand(control_dim(model))
c = rand()
w = rand(RobotDynamics.num_rotations(s))
qrefs = [rand(UnitQuaternion) for i = 1:RobotDynamics.num_rotations(s)]

## Call inner constructor
costfun = DiagonalLieCost(s, Q, R, q, r, c, w, qrefs)


# Test stage cost
TO.stage_cost(costfun, x)
p,quat,v,ω = RobotDynamics.parse_state(model, x)
Qr,Qv,Qω = Diagonal.((Q[1:3],Q[4:6],Q[7:9]))
Jv = 0.5*(p'Qr*p + v'Qv*v + ω'Qω*ω) + q[1:3]'p + q[4:6]'v + q[7:9]'ω
q0 = Rotations.params(quat)
qref = Rotations.params(qrefs[1])
dq = q0'qref
Jr = w[1]*min(1-dq,1+dq)
@test Jr + Jv ≈ TO.stage_cost(costfun, x)

Ju = 0.5*u'Diagonal(R)*u + r'u
@test TO.stage_cost(costfun, x, u) ≈ Jr + Jv + Ju

## Gradient
grad = ForwardDiff.gradient(x->TO.stage_cost(costfun, x), x)
gradq = -qref*w[1]*sign(dq)
@test grad ≈ [Qr*p + q[1:3]; gradq; Qv*v + q[4:6] ; Qω*ω + q[7:9]]
vinds = costfun.vinds
grad2 = zeros(length(grad))
grad2[vinds] .= Diagonal(Q)*x[vinds] + q

E = QuadraticCost{Float64}(13,4)
TO.gradient!(E, costfun, x, u)
@test E.q ≈ grad
@test E.r ≈ Diagonal(R)*u + r

## Hessian
hess = ForwardDiff.hessian(x->TO.stage_cost(costfun, x), x)
hess2 = zero(grad2)
hess2[vinds] .= Q
@test Diagonal(hess2) ≈ hess

E.Q .*= 0
@test TO.hessian!(E, costfun, x, u) == true
@test E.Q ≈ hess
@test E.R ≈ Diagonal(R)



## Try with an MRP
s1 = LieState(UnitQuaternion,P)
s2 = LieState(MRP,P)
dcost1 = DiagonalLieCost(s1, Qs, R)
dcost2 = DiagonalLieCost(s2, Qs, R)
model1 = Quadrotor{UnitQuaternion}()
model2 = Quadrotor{MRP}()
@test dcost2.Q == vcat(Qs...)
@test state_dim(dcost1) == 13
@test state_dim(dcost2) == 12
@test dcost2.vinds[end] == 12
@test dcost2.qinds[1] == [4,5,6]
u = @SVector rand(4)
x = rand(RBState)
x1 = RobotDynamics.build_state(model1, x)
x2 = RobotDynamics.build_state(model2, x)

@test TO.stage_cost(dcost1, x1, u) ≈ TO.stage_cost(dcost2, x2, u)

## Test conversion jacobians
q = rand(UnitQuaternion)

quat2mrp(q) = Rotations.params(MRP(UnitQuaternion(q)))
mrp2quat(g) = Rotations.params(UnitQuaternion(MRP(g)))
rp2quat(p) = Rotations.params(UnitQuaternion(RodriguesParam(p)))
quat2rp(q) = Rotations.params(RodriguesParam(UnitQuaternion(q)))

Rotations.jacobian(UnitQuaternion, MRP(q)) ≈ 
    ForwardDiff.jacobian(mrp2quat, Rotations.params(MRP(q)))
Rotations.jacobian(MRP, q) ≈ 
    ForwardDiff.jacobian(quat2mrp, Rotations.params(q))

Rotations.jacobian(UnitQuaternion, RodriguesParam(q)) ≈ 
    ForwardDiff.jacobian(rp2quat, Rotations.params(RodriguesParam(q)))

@test Rotations.jacobian(RodriguesParam, q) ≈
    ForwardDiff.jacobian(quat2rp, Rotations.params(q))

@btime Rotations.jacobian(UnitQuaternion, RodriguesParam($q))

g = Rotations.params(RodriguesParam(q))
rp2quat(g)
1/sqrt(1+g'g) * [1; g]


## Test solving a problem
model = Quadrotor()
N = 51
tf = 5.0
dt = tf / (N-1)

# Objective
x0,u0 = zeros(model) 
xf = RD.build_state(model, [2,3,1], expm(SA[0,0,1]*deg2rad(135)), zeros(3), zeros(3))
Q = [fill(1,3), fill(0.1, 6)]
R = fill(1e-2,4)
costfun = TO.LieLQRCost(RD.LieState(model), Q, R, xf)
costfun_term = TO.LieLQRCost(RD.LieState(model), Q .* 100, R, xf)
obj = Objective(costfun, costfun_term, N)
obj2 = copy(obj)

prob = Problem(model, obj, xf, tf, x0=x0)
using Altro
solver = ALTROSolver(prob, show_summary=true)
solve!(solver)
states(solver)[end]
xf_sol = RBState(model, states(solver)[end])
x̄f = RBState(model, xf)
norm(xf_sol ⊖ x̄f)