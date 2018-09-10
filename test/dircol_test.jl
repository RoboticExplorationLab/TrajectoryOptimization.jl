#***************#
#   CART POLE   #
#***************#
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
dt = 0.05


# Check Jacobians
solver = Solver(model,ConstrainedObjective(obj),dt=dt,integration=:rk3_foh)
solver.opts.verbose = true
N = solver.N
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)
xopt,uopt = solve_dircol(solver,X0,U0)


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
        TrajectoryOptimization.collocation_constraints(X,U,method,dt,solver.fc)
    end

    function eval_f(Z)
        X,U =TrajectoryOptimization. unpackZ(Z,(n,m,N))
        J = cost(solver,X,U,weights,method)
        return J
    end

    # Check constraints
    g = eval_ceq(Z)
    g_colloc = collocation_constraints(solver, res, method)
    @test g_colloc ≈ g

    Xm = res.X_
    X = res.X

    X1 = zeros(Xm)
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
    cost(solver,res) ≈ eval_f(Z)

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


# Solver integration scheme should set the dircol scheme
solver = Solver(model,obj,dt=dt,integration=:midpoint)
X,U,f,stats = solve_dircol(solver,X0,U0)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test stats["info"] == "Finished successfully: optimality conditions satisfied"

solver = Solver(model,obj,dt=dt)
X2,U2,f,stats = solve_dircol(solver,X0,U0,method=:midpoint)
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test X == X2
@test U == U2
@test stats["info"] == "Finished successfully: optimality conditions satisfied"

solver = Solver(model,obj,dt=dt,integration=:rk3_foh)
X,U,f,stats = solve_dircol(solver,X0,U0)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test stats["info"] == "Finished successfully: optimality conditions satisfied"

solver = Solver(model,obj,dt=dt)
X2,U2,f,stats = solve_dircol(solver,X0,U0,method=:hermite_simpson)
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test X == X2
@test U == U2
@test stats["info"] == "Finished successfully: optimality conditions satisfied"

# Test different derivative options
method = :hermite_simpson_separated
function check_grad_options(method)
    solver = Solver(model,obj,dt=dt)
    X,U = solve_dircol(solver,X0,U0,method=method,grads=:none)
    @test vecnorm(X[:,end]-obj.xf) < 1e-5

    X2,U2 = solve_dircol(solver,X0,U0,method=method,grads=:auto)
    @test vecnorm(X[:,end]-obj.xf) < 1e-5
    @test norm(X-X2) < 5e-3
end

check_grad_options(:hermite_simpson_separated)
check_grad_options(:hermite_simpson)
check_grad_options(:trapezoid)
check_grad_options(:midpoint)



# Mesh refinement
mesh = [0.5,0.2]
t1 = @elapsed X,U = solve_dircol(solver,X0,U0)
t2 = @elapsed X2,U2,f,stats = solve_dircol(solver,X0,U0,mesh)
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test vecnorm(X2-X) < 1e-2
@test stats["info"][end] == "Finished successfully: optimality conditions satisfied"

# No initial guess
mesh = [0.2]
t1 = @elapsed X,U = solve_dircol(solver)
t2 = @elapsed X2,U2,f,stats = solve_dircol(solver,mesh)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test vecnorm(X2-X) < 1e-2

# Test dircol constraint stuff
model, obj = Dynamics.dubinscar
obj.tf = 3
obj_con = ConstrainedObjective(obj,cE=cE,cI=cI)

method = :trapezoid
solver = Solver(model,obj_con,dt=0.1)
N, = TrajectoryOptimization.get_N(solver,method)
X = [1. 2 3 4; 1 2 3 4; 1 2 3 4]
U = [0. 1 0 0; -1 0 -1 0]
U = ones(m,N)
X = line_trajectory(obj.x0,obj.xf,N)


obj_con.p_N
pI_obj, pE_obj = TrajectoryOptimization.count_constraints(obj_con)
pI_obj == (pI, pI, 0, 0)
pE_obj == (pE, pE, pE_N+n, pE_N)
p_total = (pE + pI)*(N-1) + pE_N + pI_N

c_fun!, jac_c, lb, ub = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
@test length(ub) == length(lb) == p_total
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

# Bounds
@test lb == [zeros(3pE+pE_N); ones(3pI)*Inf]
@test ub == zeros(p_total)

# Jacobian
c_fun!, jac_c, lb, ub = TrajectoryOptimization.gen_custom_constraint_fun(solver, method)
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
@test res.fVal === res.

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
