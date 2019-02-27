using TrajectoryOptimization: generate_general_constraint_jacobian, is_inplace_function, count_inplace_output, stage_cost
using Test

""" Simple Pendulum """
model, = TrajectoryOptimization.Dynamics.pendulum
n = model.n
m = model.m
x0 = [0; 0.];
xf = [pi; 0]; # (ie, swing up)
u0 = [1]
Q = 1e-3*Diagonal(I,n)
Qf = 100. *Diagonal(I,n)
R = 1e-3*Diagonal(I,m)
tf = 5.

costfun = LQRCost(Q,R,Qf,xf)
@test_nowarn UnconstrainedObjective(costfun, tf, x0, xf)
obj_uncon = UnconstrainedObjective(costfun, tf, x0, xf)
@test obj_uncon.tf == tf

# Test minimum time constructor
obj_uncon = UnconstrainedObjective(costfun, :min, x0, xf)
@test obj_uncon.tf == 0

# Try invalid inputs
tf = -1.
@test_throws ArgumentError("tf must be non-negative") UnconstrainedObjective(costfun, tf, x0, xf)
tf = 1.; R_ = Diagonal(I,m)*-1.
@test_throws ArgumentError("R must be positive definite") LQRObjective(Q, R_, Qf, tf, x0, xf)
@test_throws ArgumentError(":min is the only recognized Symbol for the final time") obj_uncon = UnconstrainedObjective(costfun, :max, x0, xf)

function myfun(c,x,u)
    c[1:2] = x + u
end
c = zeros(2)
myfun(c,ones(2),ones(2))
is_inplace_function(myfun,ones(2),ones(2))
count_inplace_output(myfun,ones(2),ones(2))

### Constraints ###
# Test defaults
obj = ConstrainedObjective(costfun,tf,x0,xf)
@test obj.u_min == [-Inf]
@test obj.u_max == [Inf]
@test obj.x_min == -[Inf,Inf]
@test obj.x_max == [Inf,Inf]
@test isa(obj.cI(ones(0),x0,u0),Nothing)
@test isa(obj.cE(ones(0),x0,u0),Nothing)

@test obj.p == 0
@test obj.use_xf_equality_constraint == true
@test obj.p_N == 2

# Use scalar control constraints
obj = ConstrainedObjective(costfun,tf,x0,xf,u_min=-1,u_max=1)
@test obj.p == 2
@test obj.p_N == 2

# Single-sided
obj = ConstrainedObjective(costfun,tf,x0,xf,u_max=1)
@test obj.p == 1
obj = ConstrainedObjective(costfun,tf,x0,xf,u_min=1, u_max=Inf)
@test obj.p == 1

# Error testing
@test_throws ArgumentError ConstrainedObjective(costfun,tf,x0,xf,u_min=1, u_max=-1)
@test_throws DimensionMismatch ConstrainedObjective(costfun,tf,x0,xf,u_min=[1], u_max=[1,2])

# State constraints
obj = ConstrainedObjective(costfun,tf,x0,xf,x_min=-[1,2], x_max=[1,2],use_xf_equality_constraint=false)
@test obj.p == 4
@test obj.pI == 4
@test obj.pI_N == 4

@test_throws DimensionMismatch ConstrainedObjective(costfun,tf,x0,xf,x_min=-[Inf,2,3,4], x_max=[1,Inf,3,Inf])
obj = ConstrainedObjective(costfun,tf,x0,xf,x_min=-[Inf,4], x_max=[3,Inf],use_xf_equality_constraint=false)
@test obj.p == 2
@test obj.pI == 2
@test obj.pI_N == 2

# Scalar to array constraint
obj = ConstrainedObjective(costfun,tf,x0,xf,x_min=-4, x_max=4)
@test obj.p == 4
@test obj.x_max == [4,4]

# Custom constraints
function cI(cdot,x,u)
    cdot[1] = x[2]+u[1]-2
end
obj = ConstrainedObjective(costfun,tf,x0,xf,cI=cI)
@test obj.p == 1
@test obj.pI == 1
obj = ConstrainedObjective(costfun,tf,x0,xf,cE=cI)
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
@test obj.use_xf_equality_constraint == true
@test obj.p_N == 2

obj = ConstrainedObjective(obj_uncon, u_min=-1)
@test obj.p == 1

# Update objectve
obj = update_objective(obj, u_max=2, x_max = 4, cE=cI)
@test obj.p == 5

# Minimum time
c = 0.1
obj = ConstrainedObjective(costfun,tf,x0,xf, x_max=2)
@test obj.p == 2
obj = ConstrainedObjective(costfun,tf,x0,xf,u_min=-2)
@test obj.p == 1
@test obj.tf == tf
tf_ = :min
obj = ConstrainedObjective(costfun,tf_,x0,xf,u_min=-2)
@test obj.tf == 0
@test obj.p == 1
obj = ConstrainedObjective(costfun,tf_,x0,xf,u_min=-2)
@test obj.tf == 0

# Test constraint function
function cI!(cres,x,u)
    cres[1] = x[1]*x[2] + u[1]
    cres[2] = u[1]*x[2] + 3x[1]
end
function cE!(cres,x,u)
    cres[1] = x[1]^2
end
function cI!(cres,x)
    cres[1] = x[1]^2
    cres[2] = x[2]^2
end
function cE!(cres,x)
    cres[1] = x[1] + x[2] - 5
end
c = zeros(2)
cI!(c,ones(2))
is_inplace_function(cI!,ones(n))
is_inplace_function(cE!,ones(n))
count_inplace_output(cE!,ones(n))

obj = ConstrainedObjective(costfun,tf,x0,xf,u_min=-2,u_max=1,x_min=-3,x_max=4, cI=cI!, cE=cE!, cI_N=cI!, cE_N=cE!, use_xf_equality_constraint=false)
@test obj.p_N == 7
@test obj.pI_N == 6
@test obj.p == 9
@test obj.pI_custom == 2
@test obj.pE_custom == 1
@test obj.pI_N_custom == 2
@test obj.pE_N_custom == 1
@test_throws ArgumentError ConstrainedObjective(costfun,tf,x0,xf,u_min=-2,u_max=1,x_min=-3,x_max=4, cI=cI!, cE=cE!, cI_N=cI!, cE_N=cE!)
obj = ConstrainedObjective(costfun,tf,x0,xf,u_min=-2,u_max=1,x_min=-3,x_max=4, cI=cI!, cE=cE!)
@test obj.p_N == length(x0)
@test obj.pI_N == 0
@test obj.p == 9
obj = ConstrainedObjective(costfun,tf,x0,xf,u_min=-2,u_max=1,x_min=-3,x_max=4, cE=cE!)
@test obj.p_N == length(x0)
@test obj.p == 7
obj = ConstrainedObjective(costfun,tf,x0,xf,u_min=-2,u_max=1,x_min=-3,x_max=4, cE_N=cE!, use_xf_equality_constraint=false)
@test obj.p_N == 5
@test obj.p == 6

@test get_sizes(obj) == (2,1)
obj = ConstrainedObjective(costfun,tf,x0,xf,u_min=-2,u_max=1,x_min=-3,x_max=4, cI=cI!, cE=cE!)
c,c_jacob,c_labels = TrajectoryOptimization.generate_constraint_functions(obj)
cres = zeros(9)
x = [1,5]
u = [1]
cans = [0,-3,-3,1,-4,-8,6,8,1]
c(cres,x,u)
@test cres == cans

cresN = zeros(2)
c(cresN,x)
cresN
cansN = [x[1] - obj.xf[1]; x[2] - obj.xf[2]]
@test cresN == cansN
# test with minimum time
obj = ConstrainedObjective(costfun,:min,x0,xf,u_min=-2,u_max=1,x_min=-3,x_max=4, cI=cI!, cE=cE!)
@test obj.p == 9
c, c_jacob = TrajectoryOptimization.generate_constraint_functions(obj,max_dt=1.0,min_dt=0.01)
cres = zeros(12)
u_dt = [u; 0.1]
cans = [0,-0.9,-3,sqrt(0.01)-0.1,-3,1,-4,-8,6,8,1,0.1-0.9]
c(cres,[x; 0.9],u_dt)
@test cres == cans

# Change upper bound on dt
c, c_jacob = TrajectoryOptimization.generate_constraint_functions(obj,max_dt=10.)
cres = zeros(12)
cans = [0,.1-sqrt(10), -3,0  ,-3,1,-4,-8,  6,8,1,0.1-0.9]
c(cres,[x;0.9],u_dt)
@test cres == cans

# use infeasible start
u_inf = [u_dt; -1; -1]
cans = [0,.1-sqrt(10), -3,0  ,-3,1,-4,-8,  6,8,1, -1,-1, 0.1-0.9]
cres = zeros(14)
c(cres,[x;0.9],u_inf)
cres
@test cres == cans

# Verify jacobians
∇x_cI(x,u) = [x[2] x[1]; 3 u[1]]
∇u_cI(x,u) = [1; x[2]]
∇x_cE(x,u) = [2x[1] 0]
∇u_cE(x,u) = [0,]
cx = [zeros(2(m+1),n+1) ;
      Matrix(I,n,n) zeros(n);
     -Matrix(I,n,n) zeros(n);
      ∇x_cI(x,u) zeros(2);
      ∇x_cE(x,u) zeros(1);
      zeros(n,n) zeros(n);
      zeros(1,n) -1.]
cu = [Matrix(I,m+1,m+1)     zeros(m+1,n);
     -Matrix(I,m+1,m+1)     zeros(m+1,n);
      zeros(2n,m+1)         zeros(2n,n);
      ∇u_cI(x,u) zeros(2,1) zeros(2,n);
      ∇u_cE(x,u) zeros(1,1) zeros(1,n);
      zeros(n,m+1)       Matrix(I,n,n);
      zeros(1,m) 1.0 zeros(1,n)]
cx_res,cu_res = zero(cx),zero(cu)
c_jacob(cx_res,cu_res,[x; 0.9],u_inf)
@test cx_res == cx
@test cu_res == cu

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
jac_cE = generate_general_constraint_jacobian(cE,pE,n,m)
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
function _cE(xdot,x)
    xdot[1:2] = [cos(x[1]) + x[2]*x[3]; x[1]*x[2]^2]
end

pE_N = 2

jac_xN(x) = [-sin(x[1]) x[3] x[2]; x[2]^2 2x[1]*x[2] 0]

jac_cE = generate_general_constraint_jacobian(cE,pE,n,m)

_jac_cE = generate_general_constraint_jacobian(_cE,pE_N,n)
tmp1 = zeros(pE_N,n)
_jac_cE(tmp1,x)
@test tmp1 == jac_xN(x)

A1 = zeros(3,3)
B1 = zeros(3,2)
jac_cE(A1,B1,x,u)
@test A1 == jac_x(x,u)
@test B1 == jac_u(x,u)







# COST FUNCTION TESTS
using TrajectoryOptimization: taylor_expansion, stage_cost
n,m = 3,2
Q = Diagonal([1.,2,3])
R = Diagonal([4.,5])
Qf = Diagonal(ones(n)*10)
xf = Vector{Float64}(1:n)
x0 = zeros(n)

x = ones(n)
u = ones(m)*2

LinQuad = LQRCost(Q,R,Qf,xf)
J = stage_cost(LinQuad,x,u)
@test taylor_expansion(LinQuad,x,u) == (Q,R,zeros(m,n),Q*(x-xf),R*u)
@test taylor_expansion(LinQuad,x) == (Qf,Qf*(x-xf))

# Generic Cost Function
my_stage_cost(x,u) = x[1]^2 + 2*x[2]*x[3] + x[3] + 3*u[1] + u[2]^2 + u[2]*u[1] + x[2]*u[1] + log(x[3]) + sin(u[2])
my_final_cost(x) = (x[1] - 1)^2 + x[1]*x[2]
qfun(x,u) = [2*x[1], 2*x[3] + u[1], 1 + 2*x[2] + 1/x[3]]
rfun(x,u) = [3 + x[2] + u[2], 2*u[2] + u[1] + cos(u[2])]
Qfun(x,u) = [2 0 0;
          0 0 2;
          0 2 -1/x[3]^2]
Rfun(x,u) = [0 1; 1 2-sin(u[2])]
Hfun(x,u) = [0 0; 1 0; 0 0]'
qffun(x) = [2(x[1] -1) + x[2], x[1], 0]
Qffun(x) = [2x[1] 1 0; 1 0 0; 0 0 0]

mycost = GenericCost(my_stage_cost,my_final_cost,n,m)
stage_expansion = (Qfun(x,u), Rfun(x,u), Hfun(x,u), qfun(x,u), rfun(x,u))
@test taylor_expansion(mycost,x,u) == stage_expansion
@test taylor_expansion(mycost,x) == (Qffun(x), qffun(x))

mygrad(x,u) = qfun(x,u), rfun(x,u)
myhess(x,u) = Qfun(x,u), Rfun(x,u), Hfun(x,u)
myexpansion(x,u) = Qfun(x,u), Rfun(x,u), Hfun(x,u), qfun(x,u), rfun(x,u)
mygrad(x) = qffun(x)
myhess(x) = Qffun(x)
myexpansion(x) = Qffun(x), qffun(x)

mycost2 = GenericCost(my_stage_cost,my_final_cost,myexpansion,n,m)
@test taylor_expansion(mycost2,x,u) == stage_expansion
@test taylor_expansion(mycost2,x) == (Qffun(x), qffun(x))

mycost2 = GenericCost(my_stage_cost,my_final_cost,mygrad,myhess,n,m)
@test taylor_expansion(mycost2,x,u) == stage_expansion
@test taylor_expansion(mycost2,x) == (Qffun(x), qffun(x))

# Unconstrained Objective
costfun = LinQuad
obj = UnconstrainedObjective(costfun,:min,x0,xf)
@test obj.tf == 0
obj = UnconstrainedObjective(costfun,tf,x0,xf)
@test_throws ArgumentError UnconstrainedObjective(costfun,tf,x0,u)
UnconstrainedObjective(costfun,tf,x0,Float64[])
@test stage_cost(obj.cost,x,u) == J
@test stage_cost(obj,x,u) == J
