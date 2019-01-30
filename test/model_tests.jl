import TrajectoryOptimization: dynamics
using RigidBodyDynamics

f = Dynamics.dubins_dynamics!
n,m = 3,2
model = Model(f,n,m)
@test model.m == m
@test model.n == n

xdot = zeros(n)
x = rand(n)
u = rand(m)
@test evals(model) == 0
reset(model)
model.f(xdot,x,u)
@test evals(model) == 0
dynamics(model,xdot,x,u)
@test evals(model) == 1

acrobot = parse_urdf(Dynamics.urdf_doublependulum)
model = Model(acrobot)
@test evals(model) == 0
n,m = model.n, model.m
xdot = zeros(n)
x = rand(n)
u = rand(m)
dynamics(model,xdot,x,u)
@test evals(model) == 1
reset(model)
@test evals(model) == 0
