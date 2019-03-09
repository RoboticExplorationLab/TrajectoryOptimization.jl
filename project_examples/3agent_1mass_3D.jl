model = Dynamics.model_a3_m1
nb,mb = 6,9
na1,ma1 = 6,6
na2,ma2 = 6,6
na3,ma3 = 6,6
N = nb+na1+na2+na3
M = mb+ma1+ma2+ma3

bodies = (:m,:a1,:a2,:a3)
ns = (nb,na1,na2,na3)
ms = (mb,ma1,ma2,ma3)

tf = 1.0
z0 = [0.;0.;-1]
ż0 = [0.;0.;0.]
y10 = [sqrt(8/9);0.;1/3]
ẏ10 = [0.;0.;0.]
y20 = [-sqrt(2/9);sqrt(2/3);1/3]
ẏ20 = [0.;0.;0.]
y30 = [-sqrt(2/9);-sqrt(2/3);1/3]
ẏ30 = [0.;0.;0.]
x0 = [z0;ż0;y10;ẏ10;y20;ẏ20;y30;ẏ30]

zf = [10.;0.;-1]
żf = [0.;0.;0.]
y1f = [sqrt(8/9)+10;0.;1/3]
ẏ1f = [0.;0.;0.]
y2f = [-sqrt(2/9)+10;sqrt(2/3);1/3]
ẏ2f = [0.;0.;0.]
y3f = [-sqrt(2/9)+10;-sqrt(2/3);1/3]
ẏ3f = [0.;0.;0.]
xf = [zf;żf;y1f;ẏ1f;y2f;ẏ2f;y3f;ẏ3f]

Q1 = Diagonal(0.01I,nb)
R1 = Diagonal(0.000001I,mb)
Qf1 = Diagonal(1000.0I,nb)
Q2 = Diagonal(0.01I,na1)
R2 = Diagonal(0.0001I,ma1)
Qf2 = Diagonal(1000.0I,na1)
Q3 = Diagonal(0.01I,na2)
R3 = Diagonal(0.0001I,ma2)
Qf3 = Diagonal(1000.0I,na2)
Q4 = Diagonal(0.01I,na3)
R4 = Diagonal(0.0001I,ma3)
Qf4 = Diagonal(1000.0I,na3)

d = norm(z0 - y10)

function cE(c,x::AbstractArray,u)
    c[1] = norm(x[7:9] - x[1:3])^2 - d^2
    c[2] = norm(x[13:15] - x[1:3])^2 -d^2
    c[3] = norm(x[19:21] - x[1:3])^2 -d^2
    c[4] = u[1] + u[13]
    c[5] = u[2] + u[14]
    c[6] = u[3] + u[15]
    c[7] = u[4] + u[19]
    c[8] = u[5] + u[20]
    c[9] = u[6] + u[21]
    c[10] = u[7] + u[25]
    c[11] = u[8] + u[26]
    c[12] = u[9] + u[27]
end

function cE(c,x)
    c[1] = norm(x[7:9] - x[1:3])^2 - d^2
    c[2] = norm(x[13:15] - x[1:3])^2 -d^2
    c[3] = norm(x[19:21] - x[1:3])^2 -d^2
end

function ∇cE(cx,cu,x,u)
    z = x[1:3]
    y1 = x[7:9]
    y2 = x[13:15]
    y3 = x[19:21]
    cx[1,1:3] = 2(z - y1)
    cx[1,7:9] = 2(y1 - z)
    cx[2,1:3] = 2(z - y2)
    cx[2,13:15] = 2(y2 - z)
    cx[3,1:3] = 2(z - y3)
    cx[3,19:21] = 2(y3 - z)

    cu[4,1] = 1
    cu[4,13] = 1
    cu[5,2] = 1
    cu[5,14] = 1
    cu[6,3] = 1
    cu[6,15] = 1
    cu[7,4] = 1
    cu[7,19] = 1
    cu[8,5] = 1
    cu[8,20] = 1
    cu[9,6] = 1
    cu[9,21] = 1
    cu[10,7] = 1
    cu[10,25] = 1
    cu[11,8] = 1
    cu[11,26] = 1
    cu[12,9] = 1
    cu[12,27] = 1
end

function ∇cE(cx,x)
    z = x[1:3]
    y1 = x[7:9]
    y2 = x[13:15]
    y3 = x[19:21]
    cx[1,1:3] = 2(z - y1)
    cx[1,7:9] = 2(y1 - z)
    cx[2,1:3] = 2(z - y2)
    cx[2,13:15] = 2(y2 - z)
    cx[3,1:3] = 2(z - y3)
    cx[3,19:21] = 2(y3 - z)
end

cost1 = LQRCost(Q1,R1,Qf1,[zf;żf])
cost2 = LQRCost(Q2,R2,Qf2,[y1f;ẏ1f])
cost3 = LQRCost(Q3,R3,Qf3,[y2f;ẏ2f])
cost4 = LQRCost(Q4,R4,Qf4,[y3f;ẏ3f])


costs = NamedTuple{bodies}((cost1,cost2,cost3,cost4))
part_x = create_partition((nb,na1,na2,na3),bodies)
part_u = create_partition((mb,ma1,ma2,ma3),bodies)

acost = ADMMCost(costs,cE,∇cE,4,[:a1],N,M,part_x,part_u)
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

admm_plot3(res)
