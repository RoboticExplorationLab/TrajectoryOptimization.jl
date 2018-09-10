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
@test stats["info"][end] == "Finished successfully: optimality conditions satisfied"
