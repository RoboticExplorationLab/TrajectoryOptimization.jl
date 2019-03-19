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


Qm = Diagonal(1e-0I,nb)
Rm = Diagonal(1e-1I,mb)
Qfm = Diagonal(1e3I,nb)

Qa = Diagonal(1e-3I,na1)
Ra = Diagonal(1e-3I,ma1)
Qfa = Diagonal(1e1I,na1)

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
    c[2+1:N+2] = x - xf
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
    cx[2+1:N+2,:] = Diagonal(I,N)
end

cost1 = LQRCost(Qm,Rm,Qfm,[zf;żf])
cost2 = LQRCost(Qa,Ra,Qfa,[y1f;ẏ1f])
cost3 = LQRCost(Qa,Ra,Qfa,[y2f;ẏ2f])

Qm = Diagonal(1e-0I,nb)
Rm = Diagonal(1e-1I,mb)
Qfm = Diagonal(1e3I,nb)

Qa = Diagonal(1e-3I,na1)
Ra = Diagonal(1e-3I,ma1)
Qfa = Diagonal(1e1I,na1)

cost_joint = LQRCost(Diagonal([1e-0*ones(nb);1e-3*ones(na1);;1e-3*ones(na1)]),Diagonal([1e-1*ones(mb);1e-3*ones(ma1);;1e-3*ones(ma1)]),Diagonal([1e3*ones(nb);1e1*ones(na1);;1e1*ones(na1)]),xf)


costs = NamedTuple{bodies}((cost1,cost2,cost3))
part_x = create_partition((nb,na1,na2),bodies)
part_u = create_partition((mb,ma1,ma2),bodies)

acost = ADMMCost(costs,cE,∇cE,3,[:a1],N,M,part_x,part_u)
obj = UnconstrainedObjective(acost,tf,x0,xf)
obj = ConstrainedObjective(obj,cE=cE,cE_N=cE,∇cE=∇cE,use_xf_equality_constraint=false)
obj_joint = UnconstrainedObjective(cost_joint,tf,x0,xf)
obj_joint = ConstrainedObjective(obj_joint,cE=cE,cE_N=cE,∇cE=∇cE,use_xf_equality_constraint=false)

p = obj.p
p_N = obj.p_N

#Joint solve
solver = Solver(model,obj_joint,integration=:none,dt=0.1)
solver.opts.cost_tolerance = 1e-5
solver.opts.cost_tolerance_intermediate = 1e-4
solver.opts.constraint_tolerance = 1e-4
solver.opts.penalty_scaling = 2.0
solver.opts.iterations_outerloop = 30
U0 = ones(M,N-1)*5
@time res, stats = solve(solver,U0)
plot(res.X)

solver = Solver(model,obj,integration=:none,dt=0.1)
solver.opts.cost_tolerance = 1e-5
solver.opts.cost_tolerance_intermediate = 1e-4
solver.opts.constraint_tolerance = 1e-4
solver.opts.penalty_scaling = 2.0
solver.opts.iterations_outerloop = 30
res = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
U0 = ones(M,N-1)*5
@time stats = admm_solve(solver,res,U0)
admm_plot2(res)
plot(res.X)
IMAGE_DIR = joinpath(TrajectoryOptimization.root_dir(),"project_examples")
admm_plot2_start_end(res)
savefig(joinpath(IMAGE_DIR,"2i_1m_2d.png"))

plot(stats["c_max"],yscale=:log10,label="sequential",xlabel="iterations",ylabel="c_max",title="2 Double Integrators 1 Mass")



res.λ
norm(res.U[2][1:2])

res = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
@time res,stats_p = admm_solve_parallel(solver,res,U0);
admm_plot2(res)
plot!(stats_p["c_max"],label="parallel")
savefig(joinpath(IMAGE_DIR,"2i_1m_2d_c_max.png"))


res = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
initial_admm_rollout!(solver,res,ones(m,N-1)*5);
res.bodies[2:end]
res_joint =  NamedTuple{res.bodies}([copy(res) for b in res.bodies]);
agents = res.bodies[2:end]

for b in agents
    J = ilqr_solve(solver,res_joint[b],b)
    send_results!(res_joint.m,res_joint[b],b)
end
J = ilqr_solve(solver,res_joint[:m],:m)
for b in agents
    send_results!(res_joint[b],res_joint.m,:m)
end
for b in res.bodies
    update_constraints!(res_joint[b],solver)
    λ_update_default!(res_joint[b],solver)
    μ_update_default!(res_joint[b],solver)
end

c_max = max_violation(res_joint[:m])


# Parallel
res0 = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
res1 = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
res2 = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
res3 = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
initial_admm_rollout!(solver,res0,ones(M,N-1)*5)
initial_admm_rollout!(solver,res1,ones(M,N-1)*5);
initial_admm_rollout!(solver,res2,ones(M,N-1)*5);
initial_admm_rollout!(solver,res3,ones(M,N-1)*5);

ilqr_solve(solver,res0,:m)
ilqr_solve(solver,res0,:a1)
ilqr_solve(solver,res0,:a2)
to_array(res0.X)

ilqr_solve(solver,res3,:m)
send_results!(res1,res3,:m);
send_results!(res2,res3,:m);

ilqr_solve(solver,res1,:a1)
# send_results!(res2,res1,:a1);
send_results!(res3,res1,:a1);

ilqr_solve(solver,res2,:a2)
# send_results!(res1,res2,:a2);
send_results!(res3,res2,:a2);

# send_results!(res1,res3,:a1);
# send_results!(res2,res3,:a1);
# send_results!(res1,res3,:a2);
# send_results!(res2,res3,:a2);


update_constraints!(res1,solver)
update_constraints!(res2,solver)
update_constraints!(res3,solver)
λ_update_default!(res0,solver)
μ_update_default!(res0,solver)
λ_update_default!(res1,solver)
μ_update_default!(res1,solver)
λ_update_default!(res2,solver)
μ_update_default!(res2,solver)
λ_update_default!(res3,solver)
μ_update_default!(res3,solver)
println()
@show max_violation(res0)
@show max_violation(res1)
@show max_violation(res2)
@show max_violation(res3)
