# Double Integrator
T = Float64
model = TrajectoryOptimization.Dynamics.doubleintegrator
model_d = rk3(model)
n = model.n; m = model.m

# costs
Q = 1.0*Diagonal(I,n)
Qf = 1.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.]
xf = [1.; 0] # (ie, swing up)

N = 21
dt = 0.1
U0 = [0.001*rand(m) for k = 1:N-1]
obj = TrajectoryOptimization.LQRObjective(Q,R,Qf,xf,N)

u_max = 1.5
u_min = -1.5

bnd = BoundConstraint(n,m,u_max=u_max, u_min=u_min,trim=true)
goal = goal_constraint(xf)
constraints = Constraints(N)
for k = 1:N-1
    constraints[k] += bnd
end
constraints[N] += goal

doubleintegrator = TrajectoryOptimization.Problem(model_d, obj, constraints=constraints, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(doubleintegrator, U0)


model = Dynamics.DoubleIntegrator()
n,m = size(model)

# Task
x0 = @SVector [0., 0.]
xf = @SVector [1., 0]
tf = 2.0

# Discretization info
N = 21
dt = tf/(N-1)

# Costs
Q = 1.0*Diagonal(@SVector ones(n))
Qf = 1.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N)

# Constraints
u_bnd = 1.5
bnd = StaticBoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
con_bnd = ConstraintVals(bnd, 1:N-1)
conSet = ConstraintSets(n,m,[con_bnd], N)

doubleintegrator_static = StaticProblem(model, obj, xf, tf, constraints=conSet, x0=x0, N=N)
