
# model
model = TrajectoryOptimization.Dynamics.DubinsCar()
n,m = size(model)
N = 51
dt = 0.06

# cost
x0 = @SVector [0., 0., 0.]
xf = @SVector [0., 1., 0.]
tf =  3.
Qf = 100.0*Diagonal(@SVector ones(n))
Q = (1e-2)*Diagonal(@SVector ones(n))
R = (1e-2)*Diagonal(@SVector ones(m))

# constraints
u_bnd = 2.
x_min = @SVector [-0.25, -0.001, -Inf]
x_max = @SVector [0.25, 1.001, Inf]
bnd = StaticBoundConstraint(n,m,x_min=x_min,x_max=x_max,u_min=-u_bnd,u_max=u_bnd)
goal = GoalConstraint(n,m,xf)

# Constraint vals
con_bnd = ConstraintVals(bnd, 1:N-1)
con_goal = ConstraintVals(goal, N:N)

# problem
U = [@SVector ones(m) for k = 1:N-1]
obj = LQRObjective(Q,R,Qf,xf,N)

conSet = ConstraintSets(n,m,[con_bnd, con_goal], N)

parallel_park_static = Problem(model, obj, xf, tf, constraints=conSet, x0=x0, U0=U)
