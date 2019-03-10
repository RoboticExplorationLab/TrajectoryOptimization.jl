using PartedArrays
n1,m1 = 4,3
n2,m2 = 4,1
n = n1+n2
m = m1+m2

Q1 = Diagonal(1.0I,n1)
R1 = Diagonal(1.0I,m1)
Qf1 = Diagonal(0.0I,n1)
Q2 = Diagonal(1.0I,n2)
R2 = Diagonal(1.0I,m2)
Qf2 = Diagonal(10.0I,n2)

cost1 = QuadraticCost(Q1,R1,zeros(m1,n1),zeros(n1),zeros(m1),0,Qf1,zeros(n1),0)
cost2 = QuadraticCost(Q2,R2,zeros(m2,n2),zeros(n2),zeros(m2),0,Qf2,zeros(n2),0)
bodies = (:a1,:m)
costs = NamedTuple{bodies}((cost1,cost2))
costs.a1

part_x = create_partition((n1,n2),bodies)
part_u = create_partition((m1,m2),bodies)
y0 = [0.;1.;1.;0.]
v0 = zeros(m1)
z0 = [1.;0.;0.;0.]
w0 = zeros(m2)
x0 = [y0;z0]
d = 1
x = BlockArray(x0,part_x)
u = BlockArray(zeros(m1+m2),part_u)
# ϕ(c,x::BlockArray,u::BlockArray) = copyto!(c, ϕ(c,x))
# ϕ(c,x::BlockArray) = copyto!(c, norm(x[1:2] - x.[5:6])^2 - d^2)
# ϕ(c,x::Vector,u::Vector) = ϕ(c,BlockArray(x,part_x),BlockArray(u,part_u))
# ϕ(c,x::Vector) = ϕ(c,BlockArray(x,part_x))
#
# function ∇ϕ(cx,cu,x::BlockArray,u::BlockArray)
#     y = x.a1[1:2]
#     z = x.m[1:2]
#     cx[1:2] = 2(y-z)
#     cx[5:6] = -2(y-z)
# end
# ∇ϕ(cx,x) = begin y = x.a1[1:2];
#                  z = x.m[1:2]; cx[1:2] = 2(y-z); cx[5:6] = -2(y-z);  end

function cE(c,x::AbstractArray,u)
    c[1] = norm(x[1:2] - x[5:6])^2 - d^2
    c[2] = u[3] + u[5]
    c[3] = u[4] + u[6]
end
function cE(c,x)
    c[1] = norm(x[1:2] - x[5:6])^2 - d^2
end

function ∇cE(cx,cu,x,u)
    y = x[1:2]
    z = x[5:6]
    cx[1,1:2] = 2(y-z)
    cx[1,5:6] = -2(y-z)

    cu[2,3] = 1
    cu[2,5] = 1
    cu[3,4] = 1
    cu[3,6] = 1
end

function ∇cE(cx,x)
    y = x[1:2]
    z = x[5:6]
    cx[1,1:2] = 2(y-z)
    cx[1,5:6] = -2(y-z)
end

part_cx = NamedTuple{bodies}([(1:1,rng) for rng in values(part_x)])
part_cu = NamedTuple{bodies}([(1:1,rng) for rng in values(part_u)])
cx = BlockArray(zeros(1,n),part_cx)
cu = BlockArray(zeros(1,m),part_cu)

# Test joint solve
model = Dynamics.model_admm2
n1,m1 = 4,4
n2,m2 = 4,2

bodies = (:a1,:m)
ns = (n1,n2)
ms = (m1,m2)

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

Q1 = Diagonal(0.01I,n1)
R1 = Diagonal(0.000001I,m1)
Qf1 = Diagonal(1000.0I,n1)
Q2 = Diagonal(0.01I,n2)
R2 = Diagonal(0.0001I,m2)
Qf2 = Diagonal(1000.0I,n2)

cost1 = LQRCost(Q1,R1,Qf1,[yf;ẏf])#QuadraticCost(Q1,R1,zeros(m1,n1),zeros(n1),zeros(m1),0,Qf1,zeros(n1),0)
cost2 = LQRCost(Q2,R2,Qf2,[zf;żf])#QuadraticCost(Q2,R2,zeros(m2,n2),zeros(n2),zeros(m2),0,Qf2,zeros(n2),0)#LQRCost(Q2,R2,Qf2,[zf;żf])
costs = NamedTuple{bodies}((cost1,cost2))
acost = ADMMCost(costs,cE,∇cE,2,[:a1],n1+n2,m1+m2,part_x,part_u)
is_inplace_function(cE,rand(8),rand(6))
obj = UnconstrainedObjective(acost,tf,x0,xf)
obj = ConstrainedObjective(obj,cE=cE,cE_N=cE,∇cE=∇cE,use_xf_equality_constraint=false)
p = obj.p
p_N = obj.p_N

solver = Solver(model,obj,integration=:none,dt=0.1)
n,m,N = get_sizes(solver)
solver.opts.cost_tolerance = 1e-5
solver.opts.cost_tolerance_intermediate = 1e-4
solver.opts.constraint_tolerance = 1e-4
solver.opts.penalty_scaling = 2.0
res = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
U0 = zeros(model.m,solver.N-1)
J = admm_solve(solver,res,U0)

admm_plot(res)
plot(res.U,3:6)
plot(res.U,5:6)

res0 = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
res1 = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
res2 = ADMMResults(bodies,ns,ms,p,solver.N,p_N);
initial_admm_rollout!(solver,res0,rand(m,N-1)*0);
initial_admm_rollout!(solver,res1,rand(m,N-1)*0);
initial_admm_rollout!(solver,res2,rand(m,N-1)*0);

update_constraints!(res0,solver)
ilqr_solve(solver,res0,:a1)
ilqr_solve(solver,res0,:m)

update_constraints!(res1,solver)
ilqr_solve(solver,res1,:a1)

update_constraints!(res2,solver)
ilqr_solve(solver,res2,:m)
for k = 1:N
    copyto!(res2.X[k].a1, res1.X[k].a1)
    if k < N
        copyto!(res2.U[k].a1, res1.U[k].a1)
    end
end
copyto!(res2.X_,res2.X);
copyto!(res2.U_,res2.U);

for k = 1:N
    copyto!(res1.X[k].m, res2.X[k].m)
    if k < N
        copyto!(res1.U[k].m, res2.U[k].m)
    end
end
copyto!(res1.X_,res1.X);
copyto!(res1.U_,res1.U);


update_constraints!(res1,solver)
update_constraints!(res2,solver)
λ_update_default!(res0,solver)
μ_update_default!(res0,solver)
λ_update_default!(res1,solver)
μ_update_default!(res1,solver)
λ_update_default!(res2,solver)
μ_update_default!(res2,solver)
@show max_violation(res0)
@show max_violation(res1)
@show max_violation(res2)

update_constraints!(res0,solver)
update_constraints!(res1,solver)
update_jacobians!(res0,solver)
update_jacobians!(res1,solver)
to_array(res0.X) - to_array(res1.X)
Δv0 = _backwardpass_admm!(res0, solver, :a1)
Δv1 = _backwardpass_admm!(res1, solver, :a1)

cost(solver,res0)
cost(solver,res1)
rollout!(res0,solver,1.0,:a1)
rollout!(res1,solver,1.0,:a1)
cost(solver,res0,res0.X_,res0.U_)
cost(solver,res1,res1.X_,res1.U_)
J0 = forwardpass!(res0, solver, Δv0, cost(solver,res0), :a1)
J0 = forwardpass!(res1, solver, Δv1, cost(solver,res1), :a1)

to_array(res2.X) ≈ to_array(res0.X)
to_array(res2.U) ≈ to_array(res0.U)
update_constraints!(res0,solver)
update_constraints!(res2,solver)
update_jacobians!(res0,solver)
update_jacobians!(res2,solver)
Δv0 = _backwardpass_admm!(res0, solver, :m)
Δv2 = _backwardpass_admm!(res2, solver, :m)
res2.K[:m][1] ≈ res0.K[:m][1]
res2.d[:m][1] ≈ res0.d[:m][1]

cost(solver,res0)
cost(solver,res2)
rollout!(res0,solver,1.0,:m)
rollout!(res2,solver,1.0,:m)
to_array(res2.X_) ≈ to_array(res0.X_)
to_array(res2.U_)


to_array(res0.U_)
cost(solver,res0,res0.X_,res0.U_)
cost(solver,res2,res2.X_,res2.U_)
J0 = forwardpass!(res0, solver, Δv0, cost(solver,res0), :m)
J0 = forwardpass!(res2, solver, Δv2, cost(solver,res2), :m)


end
for i = 1:50

function admm_solve(solver,results,U0)
    J0 = initial_admm_rollout!(solver,res,U0);
    J = Inf
    println("J0 = $J0")
    for i = 1:50
        J = ilqr_loop(solver,res)
        println("iter: $i, J = $J")
        update_constraints!(res,solver)
        println("c_max: $(max_violation(res))")
        λ_update_default!(res,solver)
        μ_update_default!(res,solver)
    end
    return J
end

function ilqr_loop(solver::Solver,res::ADMMResults)
    J = Inf
    for b in res.bodies
        J = ilqr_solve(solver::Solver,res::ADMMResults,b::Symbol)
    end
    return J
end


function ilqr_solve(solver::Solver,res::ADMMResults,b::Symbol)
    X = res.X; U = res.U; X_ = res.X_; U_ = res.U_
    J0 = cost(solver,res)
    J = Inf

    for ii = 1:solver.opts.iterations_innerloop
        iter_inner = ii

        ### BACKWARD PASS ###
        update_constraints!(res,solver)
        update_jacobians!(res, solver)
        Δv = _backwardpass_admm!(res, solver, b)

        ### FORWARDS PASS ###
        J = forwardpass!(res, solver, Δv, J0, b)
        c_max = max_violation(res)

        ### UPDATE RESULTS ###
        copyto!(X,X_)
        copyto!(U,U_)

        dJ = copy(abs(J-J0)) # change in cost
        J0 = copy(J)

        if dJ < solver.opts.cost_tolerance
            break
        end
    end
    return J
end
