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
solver = Solver(model,obj,dt=dt,integration=:rk3_foh)
N = solver.N
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)

method = :hermite_simpson
@time solve_dircol(solver,X0,U0,method=method,grads=:auto)
@time solve_dircol(solver,X0,U0,method=method,grads=:none)

function check_grads(solver,method)

    n,m,N = get_sizes(solver)
    N = convert_N(N, method)
    U0 = ones(1,N)*1
    X0 = line_trajectory(obj.x0, obj.xf, N)

    res = DircolResults(n,m,solver.N,method)
    res.X .= X0
    res.U .= U0
    Z = res.Z
    Z[1:15] = 1:15

    weights = get_weights(method,N)*dt
    update_derivatives!(solver,res,method)
    get_traj_points!(solver,res,method)
    update_jacobians!(solver,res,method)

    function eval_ceq(Z)
        X,U = unpackZ(Z,(n,m,N))
        collocation_constraints(X,U,method,dt,solver.fc)
    end

    function eval_f(Z)
        X,U = unpackZ(Z,(n,m,N))
        J = cost(solver,X,U,weights,method)
        return J
    end

    # Check constraints
    g = eval_ceq(Z)
    g_colloc = collocation_constraints(solver, res, method)
    @test g_colloc ≈ g

    # Check constraint jacobian
    jacob_g = constraint_jacobian(solver,res,method)
    jacob_g_auto = ForwardDiff.jacobian(eval_ceq,Z)
    @test jacob_g_auto ≈ jacob_g

    # Check cost
    cost(solver,res) ≈ eval_f(Z)

    # Check cost gradient
    jacob_f_auto = ForwardDiff.gradient(eval_f,Z)
    jacob_f = cost_gradient(solver,res,method)
    @test jacob_f_auto ≈ jacob_f
end
method = :midpoint
check_grads(solver,:midpoint)
check_grads(solver,:trapezoid)
check_grads(solver,:hermite_simpson)
check_grads(solver,:hermite_simpson_separated)


solver = Solver(model,ConstrainedObjective(obj),dt=dt,integration=:rk3_foh)
solver.opts.verbose = true
method = :hermite_simpson_separated
results = DircolResults(get_sizes(solver)...,method)
usrfun = gen_usrfun(solver, results, method, grads=:none)
usrfun(results.Z)
solve_dircol(solver,X0,U0,method=method)

# Solver integration scheme should set the dircol scheme
solver = Solver(model,obj,dt=dt,integration=:midpoint)
X,U = solve_dircol(solver,X0,U0)
@test vecnorm(X[:,end]-obj.xf) < 1e-5

solver = Solver(model,obj,dt=dt)
X2,U2 = solve_dircol(solver,X0,U0,method=:midpoint)
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test X == X2
@test U == U2

solver = Solver(model,obj,dt=dt,integration=:rk3_foh)
X,U = solve_dircol(solver,X0,U0)
@test vecnorm(X[:,end]-obj.xf) < 1e-5

solver = Solver(model,obj,dt=dt)
X2,U2 = solve_dircol(solver,X0,U0,method=:hermite_simpson)
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test X == X2
@test U == U2

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
t2 = @elapsed X2,U2 = solve_dircol(solver,X0,U0,mesh)
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test vecnorm(X2-X) < 1e-2


# No initial guess
mesh = [0.2]
solver.opts.verbose = true
t1 = @elapsed X,U = solve_dircol(solver)
t2 = @elapsed X2,U2 = solve_dircol(solver,mesh)
plot(X')
plot!(X2',width=2)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test vecnorm(X2-X) < 1e-2
