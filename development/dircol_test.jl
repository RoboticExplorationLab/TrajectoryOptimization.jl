#***************#
#   CART POLE   #
#***************#
using Test
model, obj0 = Dynamics.cartpole_analytical
n,m = model.n, model.m

Q = copy(obj0.cost.Q)
R = copy(obj0.cost.R)
Qf = copy(obj0.cost.Qf)

x0 = [0;0;0;0.]
xf = [0.5;pi;0;0]
tf = 2.
u_bnd = 50
x_bnd = [0.6,Inf,Inf,Inf]
obj = LQRObjective(Q, R, Qf, tf, x0, xf)
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)
cost_mintime = LQRCost(Q*0,R*0.001,Qf,xf)
obj_mintime = update_objective(obj_con,cost=cost_mintime,tf=:min)
dt = 0.05


# Check Jacobians
method = :hermite_simpson
solver = Solver(model,ConstrainedObjective(obj),dt=dt,integration=:rk3_foh)
N = solver.N
NN = (n+m)N
nG, = get_nG(solver,method)
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)
solver.opts.verbose = true
sol,stats = solve_dircol(solver,X0,U0, method=method)
@test norm(sol.X[:,end] - obj.xf) < 1e-6
plot(sol.X')
plot(sol.U')

include("../development/dircol_old.jl")
method = :hermite_simpson
function check_grads(solver,method)
    n,m = get_sizes(solver)
    N,N_ = get_N(solver, method)
    U0 = ones(1,N)*1
    X0 = line_trajectory(obj.x0, obj.xf, N)

    res = DircolResults(n,m,solver.N,method)
    res.X .= X0
    res.U .= U0
    Z = res.Z
    Z[1:15] = 1:15

    weights = get_weights(method,N)*dt
    TrajectoryOptimization.update_derivatives!(solver,res,method)
    TrajectoryOptimization.get_traj_points!(solver,res,method)
    TrajectoryOptimization.get_traj_points_derivatives!(solver,res,method)
    TrajectoryOptimization.update_jacobians!(solver,res,method)

    function eval_ceq(Z)
        X,U = TrajectoryOptimization.unpackZ(Z,(n,m,N))
        TrajectoryOptimization.collocation_constraints(X,U,method,solver.dt,solver.fc)
    end

    function eval_f(Z)
        X,U =TrajectoryOptimization.unpackZ(Z,(n,m,N))
        cost(solver,X,U,weights,method)
    end

    # Check constraints
    g = eval_ceq(Z)
    g_colloc = collocation_constraints(solver, res, method)
    @test g_colloc ≈ g


    Xm = res.X_
    X = res.X

    X1 = zero(Xm)
    X1[:,end] = Xm[:,end]
    for k = N-1:-1:1
        X1[:,k] = 2Xm[:,k] - X1[:,k+1]
    end
    2Xm[:,N-1]-Xm[:,N]

    # Check constraint jacobian
    jacob_g = constraint_jacobian(solver,res,method)
    jacob_g_auto = ForwardDiff.jacobian(eval_ceq,Z)
    @test jacob_g_auto ≈ jacob_g

    # Check cost
    @test cost(solver,res) ≈ eval_f(Z)

    N,N_ = get_N(solver,method)
    # Check cost gradient
    jacob_f_auto = ForwardDiff.gradient(eval_f,Z)
    jacob_f = cost_gradient(solver,res,method)
    @test jacob_f_auto ≈ jacob_f
end


check_grads(solver,:midpoint)
check_grads(solver,:trapezoid)
check_grads(solver,:hermite_simpson)
check_grads(solver,:hermite_simpson_separated)


# Minimum time
method = :hermite_simpson
# Set up problem
solver = Solver(model,obj_mintime,N=51,integration=:rk3_foh)
N,N_ = TrajectoryOptimization.get_N(solver,method)
m̄, = get_num_controls(solver)
NN = N*(n+m̄)
nG,Gpart = TrajectoryOptimization.get_nG(solver,method)
# nG = Gpart.collocation
U0 = rand(1,N)*1
if is_min_time(solver)
    U0 = [U0; ones(1,N)*0.1]
end
# X0 = line_trajectory(obj.x0, obj.xf, N)
X0 = rand(n,N)
Z = TrajectoryOptimization.packZ(X0,U0)

# Init vars
g = zeros((N-1)n + N-2)
grad_f = zeros(NN)
rows = zeros(nG)
cols = zeros(nG)
vals = zeros(nG)
weights = get_weights(method,N_)

# Get functions and evaluate
eval_f, eval_g, eval_grad_f, eval_jac_g = gen_usrfun_ipopt(solver,method)

# Test Cost
eval_f(Z)
J = cost(solver::Solver,X0,U0,weights::Vector,method::Symbol)
@test eval_f(Z) ≈ J

# Cost Gradient
eval_grad_f(Z,grad_f)
grad_f
grad_f_check = ForwardDiff.gradient(eval_f,Z)
@test grad_f_check ≈ grad_f

# Constraint Jacobian
eval_g(Z,g)
eval_jac_g(Z, :Structure, rows, cols, vals)
eval_jac_g(Z, :vals, rows, cols, vals)
function eval_g_wrapper(Z)
    gin = zeros(eltype(Z),size(g))
    eval_g(Z,gin)
    return gin
end
eval_g_wrapper(Z)
jac_g_check = ForwardDiff.jacobian(eval_g_wrapper,Z)
jac_g = Array(sparse(rows,cols,vals))
@test jac_g ≈ jac_g_check






# Solver integration scheme should set the dircol scheme
U0 = ones(1,N)
solver = Solver(model,obj,dt=dt,integration=:midpoint)
sol,stats = solve_dircol(solver,X0,U0)
@test norm(sol.X[:,end]-obj.xf) < 1e-5
@test stats["info"] == :Solve_Succeeded

solver = Solver(model,obj,dt=dt)
sol2,stats = solve_dircol(solver,X0,U0,method=:midpoint)
@test norm(sol2.X[:,end]-obj.xf) < 1e-5
@test sol.X == sol2.X
@test sol.U == sol2.U
@test stats["info"] == :Solve_Succeeded

solver = Solver(model,obj,dt=dt,integration=:rk3_foh)
sol,stats = solve_dircol(solver,X0,U0)
@test norm(sol.X[:,end]-obj.xf) < 1e-5
@test stats["info"] == :Solve_Succeeded

solver = Solver(model,obj,dt=dt)
sol2,stats = solve_dircol(solver,X0,U0,method=:hermite_simpson)
@test norm(sol2.X[:,end]-obj.xf) < 1e-5
@test sol.X == sol2.X
@test sol.U == sol2.U
@test stats["info"] == :Solve_Succeeded

# No initial guess
mesh = [0.2]
t1 = @elapsed sol,stats = solve_dircol(solver)
@test norm(sol.X[:,end]-obj.xf) < 1e-5




# Test dircol constraint stuff
n,m = 3,2
cE!(c,x,u) = begin c[1:2] = 2x[1:2]+u; c[3] = x'x + 5 end
pE = 3
cE!(c,x) = begin c[1] = cos(x[1]) + x[2]*x[3]; c[2] = x[1]*x[2]^2 end
pE_N = 2
cI!(c,x,u) = begin c[1] = x[3]-x[2]; c[2] = u[1]*x[1] end
pI = 2
pI_N = 0

x = rand(n); u = rand(m);
c = zeros(3)
cI!(c,x,u)
is_inplace_function(cE!,x,u)
model, obj = Dynamics.dubinscar
obj_con = ConstrainedObjective(obj,cE=cE!,cI=cI!,cE_N=cE!,use_xf_equality_constraint=false,tf=3)
@test obj_con.p == pE + pI
@test obj_con.pI == pI
@test obj_con.pI_N == pI_N
@test obj_con.p_N == pE_N + pI_N

method = :trapezoid
solver = Solver(model,obj_con,dt=1.)
N, = TrajectoryOptimization.get_N(solver,method)
NN = (n+m)*N
X = [1. 2 3 4; 1 2 3 4; 1 2 3 4]
U = [0. 1 0 0; -1 0 -1 0]
# U = ones(m,N)
# X = line_trajectory(obj.x0,obj.xf,N)
x,u = X[:,1],U[:,1]
c = zeros(3)
@test is_inplace_function(cE!,x,u)
@test 3 == count_inplace_output(cE!,x,u)

# Constraint function
pI_obj, pE_obj = TrajectoryOptimization.count_constraints(obj_con)
@test pI_obj == (pI, pI, 0, 0)
@test pE_obj == (pE, pE, pE_N, pE_N)
p_total = (pE + pI)*(N-1) + pE_N + pI_N
p_colloc = (N-1)n

c_fun!, jac_c = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
C = zeros(p_total)
c_fun!(C,X,U)
Z = packZ(X,U)
cE(x,u) = begin c=zeros(3); cE!(c,x,u); return c end
cE(x) = begin c=zeros(2); cE!(c,x); return c end
cI(x,u) = begin c=zeros(2); cI!(c,x,u); return c end
C_expected = [cE(X[:,1],U[:,1]);
              cE(X[:,2],U[:,2]);
              cE(X[:,3],U[:,3]);
              cE(X[:,4]);
              cI(X[:,1],U[:,1]);
              cI(X[:,2],U[:,2]);
              cI(X[:,3],U[:,3])]
@test C_expected == C

x_L,x_U, g_L,g_U = get_bounds(solver,method)

# Jacobian
c_fun!, jac_c!, jac_sparsity = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
nG,Gpart = get_nG(solver,method)
nP = Gpart.custom

J_struct = jac_sparsity()
rows,cols,inds = findnz(J_struct)
v = sortperm(inds)
rows = rows[v]
cols = cols[v]

J = zeros(nP)
jac_c!(J,X,U)
J1 = Array(sparse(rows,cols,J,p_total,NN))
Z = packZ(X,U)

function c_funZ!(C,Z)
    X,U = unpackZ(Z,(n,m,N))
    c_fun!(C,X,U)
end
C2 = zeros(p_total)
c_funZ!(C2,Z)

J2 = zeros(p_total,NN)
jac_cZ!(J,Z) = ForwardDiff.jacobian!(J,c_funZ!,C2,Z)
jac_cZ!(J2,Z)
@test J2 == J1

vals = zeros(nG)
jac_c(vals,X,U)
vals


eval_f, eval_g, eval_grad_f, eval_jac_g = gen_usrfun_ipopt(solver,method)
g = zeros(p_total + p_colloc)
eval_g(Z,g)
@test g[p_colloc+1:end] == C

collocation_constraint_jacobian_sparsity(solver,method)
vals = zeros(nG)
rows = zeros(nG)
cols = zeros(nG)
eval_jac_g(Z,:Structure,rows,cols,vals)
eval_jac_g(Z,:vals,rows,cols,vals)
J_all = Array(sparse(rows,cols,vals))
J3 = J_all[p_colloc.+1:end,:]
@test J3 == J1


function eval_g_wrapper(Z)
    gin = zeros(eltype(Z),size(g))
    eval_g(Z,gin)
    return gin
end

auto_diff(Z) = ForwardDiff.jacobian(eval_g_wrapper,Z)
jac_g_check = auto_diff(Z)
@test jac_g_check == sparse(rows,cols,vals)

using BenchmarkTools
using Juno
using Profile
Profile.init(delay=1e-6)
Profile.clear()
@profile eval_jac_g(Z,:vals,rows,cols,vals)
Juno.profiler()
@btime eval_jac_g(Z,:vals,rows,cols,vals)
@btime auto_diff(Z)

@time jac_c(J,X,U)
@time jac_cZ!(J2,Z)



# DircolResults
n,m,N = (4,2,51)
res = DircolResults(n,m,N,:midpoint)
res.vars
res.Z[1] = 10.
@test res.X[1] == 10
@test res.U[1] === res.Z[5]
@test res.U[1] === res.U_[1]
@test size(res.U_) == (m,N)
@test res.vars.X == res.X
@test res.vars.Z == res.Z

res = DircolResults(n,m,N,:hermite_simpson_separated)
@test size(res.X) == (n,2N-1)
@test size(res.X_) == (n,2N-1)
@test res.X[1] === res.Z[1]
@test size(res.U) == (m,2N-1)
@test res.U === res.U_
@test res.X === res.X_

res = DircolResults(n,m,N,:hermite_simpson)
@test size(res.X) == (n,N)
@test size(res.X_) == (n,2N-1)
@test res.X[1] === res.Z[1]

res = DircolResults(n,m,N,:trapezoid)
@test res.X === res.X_

# Dircol Vars
X0 = rand(n,N)
U0 = rand(m,N)
Z0 = packZ(X0,U0)
vars = DircolVars(Z0,n,m,N)
@test X0 == vars.X
@test U0 == vars.U
@test Z0 === vars.Z
vars = DircolVars(X0,U0)
@test X0 == vars.X
@test U0 == vars.U
@test Z0 == vars.Z
