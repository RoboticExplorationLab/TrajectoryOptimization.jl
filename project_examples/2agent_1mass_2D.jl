model = Dynamics.model_a2_m1
nb,mb = 4,4
na1,ma1 = 4,4
na2,ma2 = 4,4
N = nb+na1+na2
M = mb+ma1+ma2

bodies = (:m,:a1,:a2)
ns = (nb,na1,na2)
ms = (mb,ma1,ma2)

tf = 1.0
z0 = [0.;0.]
ż0 = [0.;0.]
y10 = [-sqrt(0.5);sqrt(0.5)]
ẏ10 = [0.;0.]
y20 = [sqrt(0.5);sqrt(0.5)]
ẏ20 = [0.;0.]
x0 = [z0;ż0;y10;ẏ10;y20;ẏ20;]

zf = [10.;0.]
żf = ż0
y1f = [10.0-sqrt(0.5);sqrt(0.5)]
ẏ1f = ẏ10
y2f = [10.0+sqrt(0.5);sqrt(0.5)]
ẏ2f = ẏ20

xf = [zf;żf;y1f;ẏ1f;y2f;ẏ2f]

Q1 = Diagonal(0.01I,nb)
R1 = Diagonal(0.000001I,mb)
Qf1 = Diagonal(1000.0I,nb)
Q2 = Diagonal(0.01I,na1)
R2 = Diagonal(0.0001I,ma1)
Qf2 = Diagonal(1000.0I,na1)
Q3 = Diagonal(0.01I,na2)
R3 = Diagonal(0.0001I,ma2)
Qf3 = Diagonal(1000.0I,na2)

d = 1

function cE(c,x::AbstractArray,u)
    c[1] = norm(x[5:6] - x[1:2])^2 - d^2
    c[2] = norm(x[9:10] - x[1:2])^2 -d^2
    c[3] = u[1] + u[7]
    c[4] = u[2] + u[8]
    c[5] = u[3] + u[11]
    c[6] = u[4] + u[12]
end

function cE(c,x)
    c[1] = norm(x[1:2] - x[5:6])^2 - d^2
    c[2] = norm(x[9:10] - x[1:2])^2 -d^2
end

function ∇cE(cx,cu,x,u)
    z = x[1:2]
    y1 = x[5:6]
    y2 = x[9:10]
    cx[1,1:2] = 2(z - y1)
    cx[1,5:6] = 2(y1 - z)
    cx[2,1:2] = 2(z - y2)
    cx[2,9:10] = 2(y2 - z)

    cu[3,1] = 1
    cu[3,7] = 1
    cu[4,2] = 1
    cu[4,8] = 1
    cu[5,3] = 1
    cu[5,11] = 1
    cu[6,4] = 1
    cu[6,12] = 1
end

function ∇cE(cx,x)
    z = x[1:2]
    y1 = x[5:6]
    y2 = x[9:10]
    cx[1,1:2] = 2(z - y1)
    cx[1,5:6] = 2(y1 - z)
    cx[2,1:2] = 2(z - y2)
    cx[2,9:10] = 2(y2 - z)
end

cost1 = LQRCost(Q1,R1,Qf1,[zf;żf])
cost2 = LQRCost(Q2,R2,Qf2,[y1f;ẏ1f])
cost3 = LQRCost(Q3,R3,Qf3,[y2f;ẏ2f])

costs = NamedTuple{bodies}((cost1,cost2,cost3))
part_x = create_partition((nb,na1,na2),bodies)
part_u = create_partition((mb,ma1,ma2),bodies)

acost = ADMMCost(costs,cE,∇cE,3,[:a1],N,M,part_x,part_u)
obj = UnconstrainedObjective(acost,tf,x0,xf)
obj = ConstrainedObjective(obj,cE=cE,cE_N=cE,∇cE=∇cE,use_xf_equality_constraint=false)
p = obj.p
p_N = obj.p_N

solver = Solver(model,obj,integration=:none,dt=0.1)
solver.opts.cost_tolerance = 1e-5
solver.opts.cost_tolerance_intermediate = 1e-4
solver.opts.constraint_tolerance = 1e-4
solver.opts.penalty_scaling = 2.0
res = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
U0 = zeros(model.m,solver.N-1)
J = admm_solve(solver,res,U0)

admm_plot2(res)
