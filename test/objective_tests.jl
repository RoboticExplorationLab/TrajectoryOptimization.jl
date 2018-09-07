# include("../iLQR.jl")
# import iLQR: UnconstrainedObjective, ConstrainedObjective
using TrajectoryOptimization.Dynamics
using TrajectoryOptimization: generate_general_constraint_jacobian
using Base.Test


""" Simple Pendulum """
pendulum = Dynamics.pendulum[1]
n = pendulum.n
m = pendulum.m
x0 = [0; 0];
xf = [pi; 0]; # (ie, swing up)
u0 = [1]
Q = 1e-3*eye(n);
Qf = 100*eye(n);
R = 1e-3*eye(m);
tf = 5

@test_nowarn UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

### Constraints ###
# Test defaults
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf)
@test obj.u_min == [-Inf]
@test obj.u_max == [Inf]
@test obj.x_min == -[Inf,Inf]
@test obj.x_max == [Inf,Inf]
@test isa(obj.cI(x0,u0),Void)
@test isa(obj.cE(x0,u0),Void)
@test isa(obj.cI_N(x0),Void)
@test isa(obj.cE_N(x0),Void)
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

@test_throws DimensionMismatch obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,
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
c(x,u) = x[2]+u[1]-2
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
@test isa(obj.cI(x0,u0),Void)
@test isa(obj.cE(x0,u0),Void)
@test isa(obj.cI_N(x0),Void)
@test isa(obj.cE_N(x0),Void)
@test obj.p == 0
@test obj.use_terminal_constraint == true
@test obj.p_N == 2

obj = ConstrainedObjective(obj_uncon, u_min=-1)
@test obj.p == 1

# Update objectve
obj = update_objective(obj, u_max=2, x_max = 4, cE=c)
@test obj.p == 5


### GENERAL CONSTRAINTS ###
n,m = 3,2
cE(x,u) = [2x[1:2]+u;
          x'x + 5]
pE = 3

x = [1;2;3]
u = [1;-1]
@test cE(x,u) == [3; 3; 19]

# Jacobians
jac_cE = generate_general_constraint_jacobian(cE,pE,0,n,m)
jac_x(x,u) = [2 0 0;
           0 2 0;
           2x']
jac_u(x,u) = [1 0;
              0 1;
              0 0]
A,B = jac_cE(x,u)
@test A == jac_x(x,u)
@test B == jac_u(x,u)

# Add terminal
cE(x) = [cos(x[1]) + x[2]*x[3]; x[1]*x[2]^2]
pE_N = 2

jac_xN(x) = [-sin(x[1]) x[3] x[2];
             x[2]^2 2x[1]*x[2] 0]
jac_cE = generate_general_constraint_jacobian(cE,pE,pE_N,n,m)
@test jac_cE(x) == jac_xN(x)

A,B = jac_cE(x,u)
@test A == jac_x(x,u)
@test B == jac_u(x,u)

cI(x,u) = [x[3]-x[2]; u[1]*x[1]]
pI = 2
pI_N = 0

# Test dircol constraint stuff
model, obj = Dynamics.dubinscar
obj.tf = 3
obj_con = ConstrainedObjective(obj,cE=cE,cI=cI)

method = :trapezoid
solver = Solver(model,obj_con,dt=0.1)
N, = get_N(solver,method)
X = [1. 2 3 4; 1 2 3 4; 1 2 3 4]
U = [0. 1 0 0; -1 0 -1 0]
U = ones(m,N)
X = line_trajectory(obj.x0,obj.xf,N)


obj_con.p_N
pI_obj, pE_obj = TrajectoryOptimization.count_constraints(obj_con)
pI_obj == (pI, pI, 0, 0)
pE_obj == (pE, pE, pE_N+n, pE_N)
p_total = (pE + pI)*(N-1) + pE_N + pI_N

c_fun!, jac_c, lb, ub = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
@test length(ub) == length(lb) == p_total
C = zeros(p_total)
c_fun!(C,X,U)
Z = packZ(X,U)
C_expected = [cE(X[:,1],U[:,1]);
              cE(X[:,2],U[:,2]);
              cE(X[:,3],U[:,3]);
              cE(X[:,4]);
              cI(X[:,1],U[:,1]);
              cI(X[:,2],U[:,2]);
              cI(X[:,3],U[:,3])]
@test C_expected == C

# Bounds
@test lb == [zeros(3pE+pE_N); ones(3pI)*Inf]
@test ub == zeros(p_total)

# Jacobian
c_fun!, jac_c, lb, ub = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
J = jac_c(X,U)
Z = packZ(X,U)

function c_funZ!(C,Z)
    X,U = unpackZ(Z,(n,m,N))
    c_fun!(C,X,U)
end
C2 = zeros(p_total)
c_funZ!(C2,Z)

J2 = zeros(size(J))
jac_c!(J,Z) = ForwardDiff.jacobian!(J,c_funZ!,C2,Z)
jac_c!(J2,Z)
@test J2 == J

@time jac_c(X,U)
@time jac_c!(J2,Z)
