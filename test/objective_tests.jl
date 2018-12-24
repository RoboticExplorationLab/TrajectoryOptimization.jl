using TrajectoryOptimization: generate_general_constraint_jacobian

""" Simple Pendulum """
pendulum = Dynamics.pendulum[1]
n = pendulum.n
m = pendulum.m
x0 = [0; 0.];
xf = [pi; 0]; # (ie, swing up)
u0 = [1]
Q = 1e-3*Diagonal(I,n)
Qf = 100*Diagonal(I,n)
R = 1e-3*Diagonal(I,m)
tf = 5.

@test_nowarn UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
@test obj_uncon.c == 0
@test obj_uncon.tf == tf

# Test full constructor
obj_uncon = UnconstrainedObjective(Q, R, Qf, 0.1, tf, x0, xf)
@test obj_uncon.c == 0.1

# Test minimum time constructors
obj_uncon = UnconstrainedObjective(Q, R, Qf, 0.1, :min, x0, xf)
@test obj_uncon.c == 0.1
@test obj_uncon.tf == 0
obj_uncon = UnconstrainedObjective(Q, R, Qf, :min, x0, xf)
@test obj_uncon.c == 1.
@test obj_uncon.tf == 0

# Try invalid inputs
tf = -1.
@test_throws ArgumentError UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
tf = 1.; R_ = Diagonal(I,m)*-1
@test_throws ArgumentError("R must be positive definite") UnconstrainedObjective(Q, R_, Qf, tf, x0, xf)
c = -1.
@test_throws ArgumentError("$c is invalid input for constant stage cost. Must be positive") UnconstrainedObjective(Q,R,Qf,c,tf,x0,xf)
@test_throws ArgumentError(":min is the only recognized Symbol for the final time") obj_uncon = UnconstrainedObjective(Q, R, Qf, 0.1, :max, x0, xf)


### Constraints ###
# Test defaults
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf)
@test obj.u_min == [-Inf]
@test obj.u_max == [Inf]
@test obj.x_min == -[Inf,Inf]
@test obj.x_max == [Inf,Inf]
@test isa(obj.gs_custom(ones(0),x0),Nothing)
@test isa(obj.gc_custom(ones(0),u0),Nothing)
@test isa(obj.hs_custom(ones(0),x0),Nothing)
@test isa(obj.hc_custom(ones(0),u0),Nothing)


@test obj.use_terminal_constraint == true
@test obj.pIs == 0
@test obj.pIc == 0
@test obj.pEs == 0
@test obj.pEsN == n
@test obj.pEc == 0

# Use scalar control constraints
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,u_min=-1,u_max=1)
@test obj.pIc == 2

# Single-sided
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,u_max=1)
@test obj.pIc == 1
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,u_min=1, u_max=Inf)
@test obj.pIc == 1

# Error testing
@test_throws ArgumentError ConstrainedObjective(Q,R,Qf,tf,x0,xf,u_min=1, u_max=-1)
@test_throws DimensionMismatch ConstrainedObjective(Q,R,Qf,tf,x0,xf,u_min=[1], u_max=[1,2])

# State constraints
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,x_min=-[1,2], x_max=[1,2])
@test obj.pIs == 4

@test_throws DimensionMismatch ConstrainedObjective(Q,R,Qf,tf,x0,xf,x_min=-[Inf,2,3,4], x_max=[1,Inf,3,Inf])
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,x_min=-[Inf,4], x_max=[3,Inf])
@test obj.pIs == 2

# Scalar to array constraint
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,x_min=-4, x_max=4)
@test obj.pIs == 4
@test obj.x_max == [4,4]

# Custom constraints
function gs_custom(cdot,x)
    cdot[1] = x[1]
    cdot[2] = x[2]
end
function gc_custom(cdot,u)
    cdot[1] = u[1]
end
function hs_custom(cdot,x)
    cdot[1] = x[1]
    cdot[2] = x[2]
end
function hc_custom(cdot,u)
    cdot[1] = u[1]
end
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,gs_custom=gs_custom)
@test obj.pIs == 2
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,gc_custom=gc_custom)
@test obj.pIc == 1
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,hs_custom=hs_custom)
@test obj.pEs == 2
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,hc_custom=hc_custom)
@test obj.pEc == 1

# Construct from unconstrained
obj = ConstrainedObjective(obj_uncon)
@test obj.u_min == [-Inf]
@test obj.u_max == [Inf]
@test obj.x_min == -[Inf,Inf]
@test obj.x_max == [Inf,Inf]
@test isa(obj.gs_custom(ones(0),x0),Nothing)
@test isa(obj.gc_custom(ones(0),u0),Nothing)
@test isa(obj.hs_custom(ones(0),x0),Nothing)
@test isa(obj.hc_custom(ones(0),u0),Nothing)
@test obj.use_terminal_constraint == true
@test obj.pEsN == 2

obj = ConstrainedObjective(obj_uncon, u_min=-1)
@test obj.pIc == 1

# Update objectve
obj = update_objective(obj, u_max=2, x_max = 4, gs_custom=gs_custom)
@test obj.pIc == 2
@test obj.pIs == 4

# Minimum time
obj = ConstrainedObjective(Q,R,Qf,tf,x0,xf,x_max=2)
@test obj.pIs == 2

tf_ = :min
obj = ConstrainedObjective(Q,R,Qf,tf_,x0,xf,u_min=-2)
@test obj.tf == 0
@test obj.pIc == 1
obj = ConstrainedObjective(Q,R,Qf,tf_,x0,xf,u_min=-2)
@test obj.tf == 0

## Test constraint function
n,m = 3,2

model = Model(Dynamics.dubins_dynamics!,n,m)

# initial and goal states
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

# costs
Q = (1e-2)*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.01

function c_fun(x,u)
    x[1:n] = 10*ones(n)
end
u_max = 3
u_min = -3
x_max = 10
x_min = -10

x0 = rand(n)
u0 = rand(m)
obj_uncon = TrajectoryOptimization.UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,x_max=x_max,x_min=x_min)
@test obj_con.pIs == 2*n
@test obj_con.pIc == 2*m
@test obj_con.pEs == 0
@test obj_con.pEsN == n
@test obj_con.pEc == 0

# default state and control constraints
z = zeros(obj_con.pIs)
Z = zeros(obj_con.pIs,n)
gs = TrajectoryOptimization.generate_state_inequality_constraints(obj_con)
gsx = TrajectoryOptimization.generate_state_inequality_constraint_jacobian(obj_con)
gs(z,x0)
gsx(Z,x0)
@test isapprox(z,[x0 .- x_max; x_min .- x0])
@test isapprox(Z,[Matrix(I,n,n);-Matrix(I,n,n)])

z = zeros(obj_con.pEs)
Z = zeros(obj_con.pEs,n)
z0 = copy(z)
Z0 = copy(Z)
hs = TrajectoryOptimization.generate_state_equality_constraints(obj_con)
hsx = TrajectoryOptimization.generate_state_equality_constraint_jacobian(obj_con)
hs(z,x0)
hsx(Z,x0)
@test isapprox(z,z0)
@test isapprox(Z,Z0)

z = zeros(obj_con.pIc)
Z = zeros(obj_con.pIc,m)
gc = TrajectoryOptimization.generate_control_inequality_constraints(obj_con)
gcu = TrajectoryOptimization.generate_control_inequality_constraint_jacobian(obj_con)
gc(z,u0)
gcu(Z,u0)
@test isapprox(z,[u0 .- u_max; u_min .- u0])
@test isapprox(Z,[Matrix(I,m,m);-Matrix(I,m,m)])

z = zeros(obj_con.pEc)
Z = zeros(obj_con.pEc,m)
z0 = copy(z)
Z0 = copy(Z)

hc = TrajectoryOptimization.generate_control_equality_constraints(obj_con)
hcu = TrajectoryOptimization.generate_control_equality_constraint_jacobian(obj_con)
hc(z,u0)
hcu(Z,u0)
@test isapprox(z,z0)
@test isapprox(Z,Z0)

z = zeros(obj_con.pEsN)
Z = zeros(obj_con.pEsN,n)
hsN = TrajectoryOptimization.generate_state_terminal_constraints(obj_con)
hsNx = TrajectoryOptimization.generate_state_terminal_constraint_jacobian(obj_con)
hsN(z,x0)
hsNx(Z,x0)
@test isapprox(z,x0-xf)
@test isapprox(Z,Matrix(I,n,n))

# custom constraints
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,x_max=x_max,x_min=x_min,gs_custom=c_fun,gc_custom=c_fun,hs_custom=c_fun,hc_custom=c_fun)
@test obj_con.pIs == 2*n+n
@test obj_con.pIc == 2*m+n
@test obj_con.pEs == 0+n
@test obj_con.pEsN == n+n
@test obj_con.pEc == 0+n

z = zeros(obj_con.pIs)
Z = zeros(obj_con.pIs,n)
gs = TrajectoryOptimization.generate_state_inequality_constraints(obj_con)
gsx = TrajectoryOptimization.generate_state_inequality_constraint_jacobian(obj_con)

gs(z,x0)
gsx(Z,x0)
@test isapprox(z,[x0 .- x_max; x_min .- x0; 10*ones(n)])
@test isapprox(Z,[Matrix(I,n,n); -Matrix(I,n,n); zeros(n,n)])

z = zeros(obj_con.pEs)
z0 = copy(z)
hs = TrajectoryOptimization.generate_state_equality_constraints(obj_con)
hs(z,x0)
@test isapprox(z,10*ones(n))

Z = zeros(obj_con.pEs,n)
Z0 = copy(Z)
hsx = TrajectoryOptimization.generate_state_equality_constraint_jacobian(obj_con)
hsx(Z,x0)
@test isapprox(Z,zeros(n,n))

z = zeros(obj_con.pIc)
gc = TrajectoryOptimization.generate_control_inequality_constraints(obj_con)
gc(z,u0)
@test isapprox(z,[u0 .- u_max; u_min .- u0; 10*ones(n)])

Z = zeros(obj_con.pIc,m)
gcu = TrajectoryOptimization.generate_control_inequality_constraint_jacobian(obj_con)
gcu(Z,u0)
@test isapprox(Z,[Matrix(I,m,m);-Matrix(I,m,m);zeros(n,m)])

z = zeros(obj_con.pEc)
z0 = copy(z)
hc = TrajectoryOptimization.generate_control_equality_constraints(obj_con)
hc(z,u0)
@test isapprox(z,10*ones(n))

Z = zeros(obj_con.pEc,m)
Z0 = copy(Z)
hcu = TrajectoryOptimization.generate_control_equality_constraint_jacobian(obj_con)
hcu(Z,u0)
@test isapprox(Z,zeros(n,m))

z = zeros(obj_con.pEsN)
hsN = TrajectoryOptimization.generate_state_terminal_constraints(obj_con)
hsN(z,x0)
@test isapprox(z,[x0-xf;10*ones(n)])

Z = zeros(obj_con.pEsN,n)
hsNx = TrajectoryOptimization.generate_state_terminal_constraint_jacobian(obj_con)
hsNx(Z,x0)
@test isapprox(Z,[Matrix(I,n,n); zeros(n,n)])

# minimum time
obj_con_min_time = TrajectoryOptimization.update_objective(obj_con,tf=0.0)
@test obj_con.pIs == 2*n+n
@test obj_con.pIc == 2*m+n
@test obj_con.pEs == 0+n
@test obj_con.pEsN == n+n
@test obj_con.pEc == 0+n

z = zeros(obj_con_min_time.pIc+2)
gc = TrajectoryOptimization.generate_control_inequality_constraints(obj_con_min_time; max_dt=1.0,min_dt=1.0e-3)
gc(z,[u0; 0.1])
@test isapprox(z,[u0 .- u_max; 0.1-sqrt(1.0); u_min .- u0;sqrt(1e-3) - 0.1; 10*ones(n)])

Z = zeros(obj_con_min_time.pIc+2,m+1)
gcu = TrajectoryOptimization.generate_control_inequality_constraint_jacobian(obj_con_min_time)
gcu(Z,[u0; 0.1])
@test isapprox(Z,[Matrix(I,m+1,m+1);-Matrix(I,m+1,m+1);zeros(n,m+1)])

z = zeros(obj_con_min_time.pEc+1)
z0 = copy(z)
hc = TrajectoryOptimization.generate_control_equality_constraints(obj_con_min_time)
hc(z,[u0;0.1])
@test isapprox(z,[0;10*ones(n)])

Z = zeros(obj_con_min_time.pEc+1,m+1)
Z0 = copy(Z)
hcu = TrajectoryOptimization.generate_control_equality_constraint_jacobian(obj_con_min_time)
hcu(Z,[u0;0.1])
Z0[1,m+1] = 1.
@test isapprox(Z,Z0)

# minimum_time + infeasible
ui = [1;2;3]
U = [u0;0.1;ui]
z = zeros(obj_con_min_time.pIc+2)
gc = TrajectoryOptimization.generate_control_inequality_constraints(obj_con_min_time)
gc(z,U)
@test isapprox(z,[u0 .- u_max; 0.1-sqrt(1.0); u_min .- u0;sqrt(1e-3) - 0.1; 10*ones(n)])

Z = zeros(obj_con_min_time.pIc+2,m+1+n)
gcu = TrajectoryOptimization.generate_control_inequality_constraint_jacobian(obj_con_min_time)
gcu(Z,U)
@test isapprox(Z,[Matrix(I,m+1,m+1+n);-Matrix(I,m+1,m+1+n);zeros(n,m+1+n)])

z = zeros(obj_con_min_time.pEc+1+n)
z0 = copy(z)
hc = TrajectoryOptimization.generate_control_equality_constraints(obj_con_min_time)
hc(z,U)
@test isapprox(z,[1;2;3;0;10*ones(n)])

Z = zeros(obj_con_min_time.pEc+1+n,m+1+n)
Z0 = copy(Z)
hcu = TrajectoryOptimization.generate_control_equality_constraint_jacobian(obj_con_min_time)
hcu(Z,U)
@test isapprox(Z,[Matrix(I,n,m+1+n); 0 0 1 0 0 0; zeros(n,m+1+n)])


### GENERAL CONSTRAINTS JACOBIAN ###
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
