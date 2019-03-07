using PartedArrays
n1,m1 = 4,3
n2,m2 = 4,1

Q1 = Diagonal(1I,n1)
R1 = Diagonal(1I,m1)
Qf1 = Diagonal(10I,n1)
Q2 = Diagonal(1I,n2)
R2 = Diagonal(1I,m2)
Qf2 = Diagonal(10I,n2)

cost1 = QuadraticCost(Q1,R1,zeros(m1,n1),zeros(n1),zeros(m1),0,Qf1,zeros(n1),0)
cost2 = QuadraticCost(Q2,R2,zeros(m2,n2),zeros(n2),zeros(m2),0,Qf2,zeros(n2),0)
bodies = (:a1,:m)
costs = NamedTuple{bodies}((cost1,cost2))
costs.a1

part_x = create_partition((n1,n2),bodies)
part_u = create_partition((m1,m2),bodies)
y0 = [0.;0.;1.;0.]
v0 = zeros(m1)
z0 = [0.;0.;0.;0.]
w0 = zeros(m2)
x0 = [y0;z0]
d = 1
x = BlockArray(x0,part_x)
u = BlockArray(zeros(m1+m2),part_u)

ϕ(x::BlockArray) = norm(x.a1[1:2] - x.m[1:2]) - d^2
ϕ(x::Vector) = norm(x[part_x.a1][1:2] - x[part_x.m][1:2])^2 - d^2
function ∇ϕ(grad,x)
    y = x.a1[1:2]
    z = x.m[1:2]
    grad[1:2] = 2(y-z)
    grad[5:6] = -2(y-z)
    grad
end
∇ϕ(x) = begin grad = zeros(8); ∇ϕ(grad,x); grad end
function ∇ϕ(grad,x,b::Symbol)
    y = x.a1[1:2]
    z = x.m[1:2]
    if b == :a1
        grad[1:2] = 2(y-z)
    elseif b == :m
        grad[1:2] = -2(y-z)
    end
end
∇ϕ(x,b::Symbol) = begin grad = zeros(4); ∇ϕ(grad,x,b); grad end
ϕ(x)
ϕ(x0)
∇ϕ(x,:m)
ForwardDiff.gradient(ϕ,x0)

acost = ADMMCost(costs,ϕ,∇ϕ,2,[:a1],n1+n2,m1+m2,part_x,part_u)
stage_cost(acost,x,u)
stage_cost(cost1,y0,v0)
stage_cost(cost2,z0,w0)

taylor_expansion(acost,x,u,:m)
z0 == x.m
w0 == u.m

taylor_expansion(acost.costs.m,x.m,u.m)

ns = (n1,n2)
ms = (m1,m2)
p = 1
N = 11
res = ADMMResults(bodies,ns,ms,p,N,0);

# X  = [BlockArray(zeros(sum(ns)),part_x)   for i = 1:N];
# U  = [BlockArray(zeros(sum(ms)),part_u)   for i = 1:N-1];
#
# K  = NamedTuple{bodies}([[zeros(m,n) for i = 1:N-1] for (n,m) in zip(ns,ms)])
# d  =  NamedTuple{bodies}([[zeros(m)   for i = 1:N-1] for m in ms])
#
# testres(X,U,K,d);

## Test joint solve
model = Dynamics.model_admm
tf = 1.0
y0 = [0.;1.]
ẏ0 = [0.;0.]
z0 = [0.;0.]
ż0 = [0.;0.]
x0 = [y0;ẏ0;z0;ż0]

yf = [10.;1.]
ẏf = ẏ0
zf = [10.;0.]
żf = ż0
xf = [yf;ẏf;zf;żf]

Q = Diagonal(0.0001I,model.n)
R = Diagonal(0.0001I,model.m)
Qf = Diagonal(100.0I,model.n)

function cE(c,x::AbstractArray,u::AbstractArray)
    c[1] = norm(x[1:2] - x[5:6])^2 - d^2
    c[2] = u[3] - u[4]
end

obj = LQRObjective(Q, R, Qf, tf, x0, xf)
obj = ConstrainedObjective(obj,cE=cE,use_xf_equality_constraint=false)
solver = Solver(model,obj,integration=:none,dt=0.1)
solver.opts.verbose = true
results, stats = solve(solver,rand(model.m,solver.N-1))
plot(to_array(results.X)')
