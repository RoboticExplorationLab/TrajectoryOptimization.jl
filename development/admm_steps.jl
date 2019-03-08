model = Dynamics.model_admm
bodies = (:a1,:m)
n1,m1 = 4,3
n2,m2 = 4,1
ns = (n1,n2)
ms = (m1,m2)
n,m = sum(ns),sum(ms)
part_x = create_partition(ns,bodies)
part_u = create_partition(ms,bodies)

# Constraint
d = 1
function cE(c,x::AbstractArray)
    c[1] = norm(x[1:2] - x[5:6])^2 - d^2
    # c[2] = u[3] - u[4]
end
cE(c,x::AbstractArray,u::AbstractArray) = cE(c,x)

# Batch cost
Q = Diagonal(0.0001I,n)
R = Diagonal(0.0001I,m)
Qf = Diagonal(100.0I,n)

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
obj_b = LQRObjective(Q,R,Qf,tf,x0,xf)
obj_b = ConstrainedObjective(obj_b, tf=tf,use_xf_equality_constraint=false,cE=cE,cE_N=cE)

# ADMM Cost
Q1 = Diagonal(Array(Q[part_x.a1,part_x.a1]))
R1 = Diagonal(Array(R[part_u.a1,part_u.a1]))
Qf1 = Diagonal(Array(Qf[part_x.a1,part_x.a1]))
Q2 = Diagonal(Array(Q[part_x.m,part_x.m]))
R2 = Diagonal(Array(R[part_u.m,part_u.m]))
Qf2 = Diagonal(Array(Qf[part_x.m,part_x.m]))

cost1 = LQRCost(Q1,R1,Qf1,[yf;ẏf])
cost2 = LQRCost(Q2,R2,Qf2,[zf;żf])
costs = NamedTuple{bodies}((cost1,cost2))
acost = ADMMCost(costs,null_constraint,null_constraint_jacobian,2,[:a1],n1+n2,m1+m2,part_x,part_u)
obj_a = ConstrainedObjective(acost,tf,x0,xf,use_xf_equality_constraint=false,cE=cE,cE_N=cE)


solver_b = Solver(model,obj_b,integration=:none,dt=0.1)
solver_b.opts.verbose = true
solver_b.opts.cost_tolerance = 1e-8
results, stats = solve(solver_b)
plot(to_array(results.U)')
