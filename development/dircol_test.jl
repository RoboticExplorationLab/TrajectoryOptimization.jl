#***************#
#   CART POLE   #
#***************#
using Test
model, obj0 = Dynamics.cartpole_analytical
n,m = model.n, model.m

obj = copy(obj0)
obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 50
x_bnd = [0.6,Inf,Inf,Inf]
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)
obj_mintime = update_objective(obj_con,tf=:min,Q=obj.Q*0,Qf=obj.Qf,R=obj.R*0.001)
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

# Solve minimum time
method = :hermite_simpson
solver = Solver(model,obj_mintime,N=41,integration=:rk3_foh)
n,m,N = get_sizes(solver)
m̄, = get_num_controls(solver)
nG, = get_nG(solver,method)
NN = (n+m̄)N
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)
# U0 = Matrix(sol.U)
# X0 = Matrix(sol.X)
U0_dt = [U0;ones(1,N)*0.05]
solver.opts.verbose = true
solver.opts.R_minimum_time = 100

sol, = solve_ipopt(solver, X0, U0_dt, method)
@test norm(sol.X[:,end] - obj.xf) < 1e-6
plot(sol.X')
# sol,stats = solve_dircol(solver,X0,U0_dt)



Z0 = packZ(X0,U0_dt)
# Generate functions
eval_f, eval_g, eval_grad_f, eval_jac_g = gen_usrfun_ipopt(solver,method)

# Get Bounds
x_L, x_U, g_L, g_U = get_bounds(solver,method)
P = length(g_L)  # Total number of constraints

all(Z0 .>= x_L)
all(Z0 .<= x_U)

# Test functions
g = zeros(P)
grad_f = zeros(NN)
jac_g = zeros(nG)
rows = zeros(nG)
cols = zeros(nG)
eval_f(Z0)
eval_g(Z0,g)
eval_grad_f(Z0,grad_f)
eval_jac_g(Z0,:Structure,rows,cols,nG)
eval_jac_g(Z0,:vals,rows,cols,jac_g)


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
cE(x,u) = [2x[1:2]+u;
          x'x + 5]
pE = 3
cE(x) = [cos(x[1]) + x[2]*x[3]; x[1]*x[2]^2]
pE_N = 2
cI(x,u) = [x[3]-x[2]; u[1]*x[1]]
pI = 2
pI_N = 0

model, obj = Dynamics.dubinscar
obj.tf = 3
obj_con = ConstrainedObjective(obj,cE=cE,cI=cI)

method = :trapezoid
solver = Solver(model,obj_con,dt=1.)
N, = TrajectoryOptimization.get_N(solver,method)
X = [1. 2 3 4; 1 2 3 4; 1 2 3 4]
U = [0. 1 0 0; -1 0 -1 0]
# U = ones(m,N)
# X = line_trajectory(obj.x0,obj.xf,N)


obj_con.p_N
pI_obj, pE_obj = TrajectoryOptimization.count_constraints(obj_con)
@test pI_obj == (pI, pI, 0, 0)
@test pE_obj == (pE, pE, pE_N+n, pE_N)
p_total = (pE + pI)*(N-1) + pE_N + pI_N

c_fun!, jac_c = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
C = zeros(p_total)
c_fun!(C,X,U)
Z = packZ(X,U)
C_expected = [cE(X[:,1],U[:,1]);
              cE(X[:,2],U[:,2]);
              cE(X[:,3],U[:,3]);
              cE(X[:,4]);
              cI(X[:,1],U[:,1]);
              cI(X[:,2],U[:,2]);
              cI(X[:,3],U[:,3])]
@test C_expected == C

# Jacobian
c_fun!, jac_c = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
J = jac_c(X,U)
Z = packZ(X,U)

function c_funZ!(C,Z)
    X,U = unpackZ(Z,(n,m,N))
    c_fun!(C,X,U)
end
C2 = zeros(p_total)
c_funZ!(C2,Z)

J2 = zeros(size(J))
jac_c!(J,Z) = ForwardDiff.jacobian!(J,c_funZ!,C2,Z)
jac_c!(J2,Z)
@test J2 == J

@time jac_c(X,U)
@time jac_c!(J2,Z)

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
