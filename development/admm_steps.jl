model = Dynamics.model_admm2
bodies = (:a1,:m)
n1,m1 = 4,4
n2,m2 = 4,2
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
function ∇cE(cx,cu,x,u)
    y = x[1:2]
    z = x[5:6]
    cx[1:2] = 2(y-z)
    cx[5:6] = -2(y-z)
end
function ∇cE(cx,x)
    y = x[1:2]
    z = x[5:6]
    cx[1:2] = 2(y-z)
    cx[5:6] = -2(y-z)
end


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
obj_a = ConstrainedObjective(acost,tf,x0,xf,use_xf_equality_constraint=false,cE=cE,cE_N=cE, ∇cE=∇cE)

cost1 = LQRCost(Q1,R1,Qf1,xf)
costs = NamedTuple{bodies}((cost1,))
acost = ADMMCost(costs,null_constraint,null_constraint_jacobian,1,[:a1],n1+n2,m1+m2,part_x,part_u)
obj_a = ConstrainedObjective(acost,tf,x0,xf,use_xf_equality_constraint=false,cE=cE,cE_N=cE,  ∇cE=∇cE)

solver_b = Solver(model,obj_b,integration=:none,dt=0.1)
solver_a = Solver(model,obj_a,integration=:none,dt=0.1)
n,m,N = get_sizes(solver_b)
p,pE,pI = get_num_constraints(solver_a)
p_N,pE_N,pI_N = get_num_terminal_constraints(solver_a)
b = :a1

X0 = Array{Float64,2}(undef,0,0)
U0 = rand(m,N-1)
res_b = init_results(solver_b,X0,U0)
res_a = ADMMResults(bodies,ns,ms,p,N,p_N);
rollout!(res_b,solver_b)
initial_admm_rollout!(solver_a,res_a,U0)
update_constraints!(res_b,solver_b)
update_constraints!(res_a,solver_a)

Jb = cost(solver_b,res_b)
Ja = cost(solver_a,res_a)

update_jacobians!(res_b,solver_b)
update_jacobians!(res_a,solver_a)
res_b.fdx[N-1] ≈ res_a.fdx[N-1]
res_b.fdu[N-1] ≈ res_a.fdu[N-1]
res_b.Cx[N-1] ≈ res_a.Cx[N-1]
res_b.Cu[N-1] == res_a.Cu[N-1]

Δvb = backwardpass!(res_b,solver_b)
Δva = _backwardpass_admm!(res_a, solver_a, :a1)
res_b.S[N][1:4,1:4] ≈ res_a.S[:a1][N]
forwardpass!(res_b,solver_b,Δvb,Jb)
forwardpass!(res_a,solver_a,Δva,Ja,b)

copyto!(res_b.X,res_b.X_)
copyto!(res_b.U,res_b.U_)
copyto!(res_a.X,res_a.X_);
copyto!(res_a.U,res_a.U_);
to_array(res_b.X) ≈ to_array(res_a.X)
to_array(res_b.U) ≈ to_array(res_a.U)

max_violation(res_b)
max_violation(res_a)

p = plot(res_b.U,1:2)
plot!(to_array(res_a.U)[1:2,:]');
display(p)

outer_loop_update(res_a,solver_a)
outer_loop_update(res_b,solver_b)
to_array(res_b.λ) ≈ to_array(res_a.λ)
