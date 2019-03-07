# Parameters
bodies = (:a1,:m)
n1,m1 = 4,2
n2,m2 = 4,0
ns = (n1,n2)
ms = (m1,m2)
n = sum(ns)
m = sum(ms)
p = 1
N = 11
tf = 2.
d = 1   # length of rod

part_x = create_partition((n1,n2),bodies)
part_u = create_partition((m1,m2),bodies)

# Dynamics (temp)
f_d(xdot,x,u) = copyto!(xdot,2*x)
model = Model(f_d,n,m)

# Initial and final states
y0 = [0,1.,0,0]
v0 = zeros(m1)
z0 = [0.5,0,0,0]
w0 = zeros(m2)
x0 = [y0;z0]
u0 = zeros(m1+m2)
xf = [1,1,0,0,1,0,0,0]

# Cost functions
Q1 = Diagonal(1I,n1)
R1 = Diagonal(1I,m1)
Qf1 = Diagonal(10I,n1)
Q2 = Diagonal(1I,n2)
R2 = Diagonal(1I,m2)
Qf2 = Diagonal(10I,n2)

cost1 = QuadraticCost(Q1,R1,zeros(m1,n1),zeros(n1),zeros(m1),0,Qf1,zeros(n1),0)
cost2 = QuadraticCost(Q2,R2,zeros(m2,n2),zeros(n2),zeros(m2),0,Qf2,zeros(n2),0)
costs = NamedTuple{bodies}((cost1,cost2))

# Create test arrays
x = BlockArray(x0,part_x)
u = BlockArray(u0,part_u)

# Constraint function
ϕ(x::BlockArray,u::BlockArray) = norm2(x.a1[1:2] - x.m[1:2]) - d^2
ϕ(x::Vector,u::Vector) = norm(x[part_x.a1][1:2] - x[part_x.m][1:2])^2 - d^2
function ∇ϕ(cx,cu,x::BlockVector,u::BlockVector)
    y = x.a1[1:2]
    z = x.m[1:2]
    cx[1,1:2] = 2(y-z)
    cx[1,5:6] = -2(y-z)
end
∇ϕ(x::BlockVector,u::BlockVector) = begin
    cx,cu = zeros(1,8), zeros(1,2); ∇ϕ(cx,cu,x,u); grad end
function ∇ϕ(cx,cu,x::BlockVector,u::BlockVector,b::Symbol)
    y = x.a1[1:2]
    z = x.m[1:2]
    if b == :a1
        cx[1,1:2] = 2(y-z)
    elseif b == :m
        cx[1,1:2] = -2(y-z)
    end
end
∇ϕ(x::BlockVector,u::BlockVector,b::Symbol) = begin
    cx,cu = zeros(1,8), zeros(1,2); ∇ϕ(cx,cu,x,u,b); cx,cu end
ϕ(x,u)
ϕ(x0,u0)
cx = BlockArray(zeros(1,n),NamedTuple{bodies}([(1:1,rng) for rng in values(part_x)]))
cu = BlockArray(zeros(1,m),NamedTuple{bodies}([(1:1,rng) for rng in values(part_u)]))
∇ϕ(cx,cu,x,u)


acost = ADMMCost(costs,ϕ,∇ϕ,2,[:a1],n,m,part_x,part_u)
stage_cost(acost,x,u)
stage_cost(cost1,y0,v0)
stage_cost(cost2,z0,w0)
taylor_expansion(acost,x,u,:a1)


cE(c,x,u) = copyto!(c,ϕ(x,u))
obj = ConstrainedObjective(acost,tf,x0,xf,cE=cE,∇cE=∇ϕ)
solver = Solver(model,obj,N=N,integration=:none)
res = ADMMResults(bodies,ns,ms,p,N,obj.p_N);

X = [x for k = 1:N];
U = [u for k = 1:N-1];
copyto!(res.X,X);
copyto!(res.U,U);
_cost(solver,res)
update_constraints!(res,solver)
update_jacobians!(res,solver)
res.fdx[1][:a1]
res.Cx[N].a1

X  = [BlockArray(zeros(sum(ns)),part_x)   for i = 1:N];
U  = [BlockArray(zeros(sum(ms)),part_u)   for i = 1:N-1];

K  = NamedTuple{bodies}([[zeros(m,n) for i = 1:N-1] for (n,m) in zip(ns,ms)])
d  =  NamedTuple{bodies}([[zeros(m)   for i = 1:N-1] for m in ms])

testres(X,U,K,d);
