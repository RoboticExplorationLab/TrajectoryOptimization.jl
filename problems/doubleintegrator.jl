
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
bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
con_bnd = ConstraintVals(bnd, 1:N-1)
conSet = ConstraintSets(n,m,[con_bnd], N)

doubleintegrator_static = Problem(model, obj, xf, tf, constraints=conSet, x0=x0, N=N)
