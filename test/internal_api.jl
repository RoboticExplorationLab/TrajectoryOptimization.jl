using TrajectoryOptimization
using RobotDynamics
using StaticArrays, LinearAlgebra
using Rotations
using RobotZoo: Quadrotor
using Test
using BenchmarkTools
using ForwardDiff
const TO = TrajectoryOptimization
const RD = RobotDynamics

## Create a Problem
model = Quadrotor()
n,m = size(model)           # number of states and controls
n̄ = RD.errstate_dim(model)  # size of error state
N = 51                      # number of knot points
tf = 5.0                    # final time

# initial and final conditions
x0 = RBState([1,2,1], UnitQuaternion(I), zeros(3), zeros(3))
xf = RBState([0,0,2], UnitQuaternion(I), zeros(3), zeros(3))

# objective
Q = Diagonal(@SVector fill(0.1, n))
R = Diagonal(@SVector fill(0.01, m))
Qf = Diagonal(@SVector fill(100.0, n))
obj = LQRObjective(Q,R,Qf,xf,N)
obj = LQRObjective(Q,R,Qf,xf,N, diffmethod=RD.UserDefined())

# constraints
cons = ConstraintList(n,m,N)
add_constraint!(cons, BoundConstraint(n,m, u_min=zeros(4), u_max=fill(10.0,4)), 1:N-1)
add_constraint!(cons, CircleConstraint(n, SA_F64[1,2], SA_F64[1,2], SA[0.1,0.1]), 1:N-1)
add_constraint!(cons, GoalConstraint(xf, SA[1,2,3]), N)

# problem
prob = Problem(model, obj, x0, tf, xf=xf, constraints=cons);
Z = prob.Z

## Initialize the controls
u0 = zeros(model)[2]                        # get hover control
initial_controls!(prob, u0)                 # set all time-steps to the same
initial_controls!(prob, [u0 for k = 1:N-1]) # use a vector of initial controls
initial_controls!(prob, fill(u0[1],m,N-1))  # use a matrix of initial controls

## Simulating the Dynamics
using RobotDynamics: state, control, states, controls
# simulate the system forward
rollout!(prob)
@test state(prob.Z[1]) == prob.x0
@test states(prob)[end] ≈ prob.x0

# alternative method
rollout!(RD.StaticReturn(), prob.model, Z, prob.x0)
@test state(prob.Z[1]) == prob.x0
@test states(prob)[end] ≈ prob.x0

# change control so that the state changes
u0 += [1,0,1,0]*1e-2
initial_controls!(prob, u0)
RD.rollout!(RD.StaticReturn(), prob.model, Z, prob.x0)
states(prob)[end]

Zmut = Traj([KnotPoint{n,m}(MVector(z.z), z.t, z.dt) for z in Z])
RD.rollout!(RD.InPlace(), prob.model, Zmut, prob.x0)


## Computing the dynamics Jacobians
D = [TO.DynamicsExpansion{Float64}(n,n̄,m) for k = 1:N-1]
TO.dynamics_expansion!(RD.StaticReturn(), RD.ForwardAD(), prob.model, D, Z)
TO.dynamics_expansion!(RD.StaticReturn(), RD.FiniteDifference(), prob.model, D, Z)
TO.dynamics_expansion!(RD.InPlace(), RD.ForwardAD(), prob.model, D, Z)
TO.dynamics_expansion!(RD.InPlace(), RD.FiniteDifference(), prob.model, D, Z)

G = [SizedMatrix{n,n̄}(zeros(n,n̄)) for k = 1:N+1]
RD.state_diff_jacobian!(prob.model, G, Z)
TO.error_expansion!(D, prob.model, G)
A,B = TO.error_expansion(D[1], prob.model)

## Computing the cost
cost(prob)

TO.cost!(prob.obj, Z)

TO.get_J(obj)

RD.evaluate(obj[1], Z[1])

## Cost Expansion
E0 = TO.CostExpansion(n, m, N)
TO.cost_expansion!(E0, prob.obj, Z)

TO.cost_gradient!(E0, prob.obj, Z)
TO.cost_hessian!(E0, prob.obj, Z)

RD.gradient!(obj.diffmethod[1], obj[1], E0[1].grad, Z[1])
RD.hessian!(obj.diffmethod[1], obj[1], E0[1].hess, Z[1])

E = TO.CostExpansion(n̄,m,N)
E0 = TO.CostExpansion(E, prob.model)
TO.error_expansion!(E, E0, prob.model, Z, G)

## Constraints
con,inds = cons[2], 1:N-1
@show typeof(con)                     # just verifying it's a CircleConstraint
@test con isa TO.StateConstraint      # it inherits from StateConstraint, so it's a function of a single state
@test state_dim(con) == n             # the state dimension. control_dim won't be defined.
@test TO.check_dims(con, n, m)        # useful method to check if a constraint is consistent with the problem sizes
p = length(con);                      # get the length of the constraint vector
println("Length of the constraint: $p")

c = zeros(length(con))
RD.evaluate(con, Z[1])
RD.evaluate!(con, c, Z[1])

vals = [zero(c) for i = inds]
jacs0 = [zeros(length(con), n+m) for i in inds]
jacs = [zeros(length(con), n̄+m) for i in inds]

RD.evaluate!(RD.StaticReturn(), con, vals, Z, inds)
@test vals[1] ≈ RD.evaluate(con, Z[1])

RD.evaluate!(RD.InPlace(), con, vals, Z, inds)
@test vals[1] ≈ RD.evaluate(con, Z[1])

RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), con, jacs0, vals, Z, inds)
f(z) = RD.evaluate(con, z[1:n], z[n+1:end])
@test jacs0[1] ≈ ForwardDiff.jacobian(f, RD.getdata(Z[1]))

TO.error_expansion!(jacs, jacs0, con, prob.model, G, inds)

## Dynamics Constraints
dyn = TO.DynamicsConstraint(prob.model)

vals = [zeros(n) for k = 1:N]
jacs = [zeros(n,n+m) for k = 1:N, i = 1:2]
RD.evaluate!(RD.StaticReturn(), dyn, vals, Z)
RD.evaluate!(RD.InPlace(), dyn, vals, Z)

RD.jacobian!(RD.StaticReturn(), RD.FiniteDifference(), dyn, jacs, vals, Z)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), dyn, jacs, vals, Z)
@test jacs[1,1] ≈ D[1].∇f
@test jacs[1,2] ≈ [-I(n) zeros(n,m)]

# Implicit dynamics
model_implicit = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
dyn_implicit = TO.DynamicsConstraint(model_implicit)

RD.evaluate!(RD.StaticReturn(), dyn_implicit, vals, Z)
RD.evaluate!(RD.InPlace(), dyn_implicit, vals, Z)

RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dyn_implicit, jacs, vals, Z)
RD.jacobian!(RD.InPlace(), RD.FiniteDifference(), dyn_implicit, jacs, vals, Z)