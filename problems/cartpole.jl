const TO = TrajectoryOptimization
import TrajectoryOptimization: KnotPoint, StaticBoundConstraint, GoalConstraint, StaticProblem, ConstraintVals, ConstraintSets
using StaticArrays

# Cartpole
T = Float64
model = TrajectoryOptimization.Dynamics.cartpole
model_d = rk3(model)
n = model.n; m = model.m

# costs
Q = 1.0e-2*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.; 0.; 0.]
xf = [0.; pi; 0.; 0.] # (ie, swing up)

N = 101
tf = 5.
dt = tf/(N-1)
u0 = 0.01*ones(m)
U0 = [u0 for k = 1:N-1]
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)
u_bnd = 3.0
bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
goal = goal_constraint(xf)
constraints = Constraints(N)
for k = 1:N-1
    constraints[k] += bnd
end
constraints[N] += goal

cartpole = TrajectoryOptimization.Problem(model_d, obj, constraints=constraints,x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(cartpole, U0)


model = Dynamics.Cartpole()
# TO.generate_jacobian(model)
# TO.rk3_gen(model)
# TO.generate_discrete_jacobian(model)

Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
x0 = SVector{n}(x0)
xf = SVector{n}(xf)
obj = LQRObjective(Q,R,Qf,xf,N)

xs = NaN*@SVector zeros(n)
us = SVector{m}(u0)
Z = [KnotPoint(xs,us,dt) for k = 1:N]
Z[end] = KnotPoint(xs,m)

bnd = StaticBoundConstraint(n,m, u_min=-u_bnd*(@SVector ones(m)), u_max=u_bnd*(@SVector ones(m)))
goal = GoalConstraint(SVector{n}(xf))
con_bnd = ConstraintVals(bnd, 1:N-1)
con_goal = ConstraintVals(goal, N:N)
conSet = ConstraintSets([con_bnd, con_goal], N)

cartpole_static = StaticProblem(model, obj, conSet, x0, xf,
    deepcopy(Z), deepcopy(Z), N, dt, dt*(N-1))
