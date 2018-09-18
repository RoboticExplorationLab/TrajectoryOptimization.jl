using TrajectoryOptimization: generate_general_constraint_jacobian



""" Simple Pendulum """
pendulum = Dynamics.pendulum[1]
n = pendulum.n
m = pendulum.m
x0 = [0; 0.];
xf = [pi; 0]; # (ie, swing up)
u0 = [1]
Q = 1e-3*Diagonal(I,n);
Qf = 100*Diagonal(I,n)
R = 1e-3*Diagonal(I,m)
tf = 5.

@test_nowarn UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

### Constraints ###
# Test defaults
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf)
@test obj.u_min == [-Inf]
@test obj.u_max == [Inf]
@test obj.x_min == -[Inf,Inf]
@test obj.x_max == [Inf,Inf]
@test isa(obj.cI(ones(0),x0,u0),Nothing)
@test isa(obj.cE(ones(0),x0,u0),Nothing)

@test obj.p == 0
@test obj.use_terminal_constraint == true
@test obj.p_N == 2

# Use scalar control constraints
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    u_min=-1,u_max=1)
@test obj.p == 2
@test obj.p_N == 2

# Single-sided
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    u_max=1)
@test obj.p == 1
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    u_min=1, u_max=Inf)
@test obj.p == 1

# Error testing
@test_throws ArgumentError ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    u_min=1, u_max=-1)
@test_throws DimensionMismatch ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    u_min=[1], u_max=[1,2])

# State constraints
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    x_min=-[1,2], x_max=[1,2])
@test obj.p == 4
@test obj.pI == 4

@test_throws DimensionMismatch ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    x_min=-[Inf,2,3,4], x_max=[1,Inf,3,Inf])
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    x_min=-[Inf,4], x_max=[3,Inf])
@test obj.p == 2
@test obj.pI == 2

# Scalar to array constraint
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    x_min=-4, x_max=4)
@test obj.p == 4


# Custom constraints
function c(cdot,x,u)
    cdot[1] = x[2]+u[1]-2
end
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    cI=c)
@test obj.p == 1
@test obj.pI == 1
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
    cE=c)
@test obj.p == 1
@test obj.pI == 0

# Construct from unconstrained
obj = ConstrainedObjective(obj_uncon)
@test obj.u_min == [-Inf]
@test obj.u_max == [Inf]
@test obj.x_min == -[Inf,Inf]
@test obj.x_max == [Inf,Inf]
@test isa(obj.cI(ones(0),x0,u0),Nothing)
@test isa(obj.cE(ones(0),x0,u0),Nothing)
@test obj.p == 0
@test obj.use_terminal_constraint == true
@test obj.p_N == 2

obj = ConstrainedObjective(obj_uncon, u_min=-1)
@test obj.p == 1

# Update objectve
obj = update_objective(obj, u_max=2, x_max = 4, cE=c)
@test obj.p == 5


### GENERAL CONSTRAINTS ###
n, m = 3,2
function cE(cdot,x,u)
     cdot[1:2] = 2x[1:2]+u
     cdot[3] = x'x + 5
end
pE = 3

x = [1;2;3]
u = [1;-1]
cdot = zeros(3)
cE(cdot,x,u)
@test cdot == [3; 3; 19]

# Jc = zeros(pE,n+m)
# dS = zeros(pE)
# S = zeros(n+m)
# c_aug! = f_augmented!(cE,n,m)
# Fc!(Jc,dS,S) = ForwardDiff.jacobian!(Jc,c_aug!,dS,S)
# Fc!(Jc,dS,S)


# Jacobians
jac_cE = generate_general_constraint_jacobian(cE,pE,0,n,m)
jac_x(x,u) = [2 0 0;
              0 2 0;
              2x']
jac_u(x,u) = [1 0;
              0 1;
              0 0]
A = zeros(3,3)
B = zeros(3,2)
jac_cE(A,B,x,u)
@test A == jac_x(x,u)
@test B == jac_u(x,u)

# Add terminal
function cE(xdot,x)
    xdot[1:2] = [cos(x[1]) + x[2]*x[3]; x[1]*x[2]^2]
end

pE_N = 2

jac_xN(x) = [-sin(x[1]) x[3] x[2]; x[2]^2 2x[1]*x[2] 0]

jac_cE = generate_general_constraint_jacobian(cE,pE,pE_N,n,m)
tmp1 = zeros(pE_N,n)
jac_cE(tmp1,x)
@test tmp1 == jac_xN(x)

A1 = zeros(3,3)
B1 = zeros(3,2)
jac_cE(A1,B1,x,u)
@test A1 == jac_x(x,u)
@test B1 == jac_u(x,u)

# cI(x,u) = [x[3]-x[2]; u[1]*x[1]]
# pI = 2
# pI_N = 0
