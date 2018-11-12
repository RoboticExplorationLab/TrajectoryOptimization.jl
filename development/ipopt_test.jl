using TrajectoryOptimization: get_nG, gen_usrfun_ipopt, update_derivatives!, get_traj_points!, get_traj_points_derivatives!, update_jacobians!
model, obj0 = Dynamics.cartpole_analytical
obj = copy(obj0)
n,m = model.n, model.m
dt = 0.1


function test_ipopt_funcs(method)
    method = :trapezoid
    # Set up problem
    solver = Solver(model,ConstrainedObjective(obj),dt=dt,integration=:rk3_foh)
    N,N_ = TrajectoryOptimization.get_N(solver,method)
    NN = N*(n+m)
    nG = TrajectoryOptimization.get_nG(solver,method)
    U0 = ones(1,N_)*1
    X0 = line_trajectory(obj.x0, obj.xf, N_)
    Z = TrajectoryOptimization.packZ(X0,U0)

    # Init vars
    g = zeros((N-1)n)
    grad_f = zeros(NN)
    rows = zeros(nG)
    cols = zeros(nG)
    vals = zeros(nG)

    # Get functions and evaluate
    eval_f, eval_g, eval_grad_f, eval_jac_g = gen_usrfun_ipopt(solver,method)
    J = eval_f(Z)
    eval_g(Z,g)
    eval_grad_f(Z,grad_f)
    eval_jac_g(Z,:Structure,rows,cols,vals)
    eval_jac_g(Z,:vals,rows,cols,vals)
    jac_g = sparse(rows,cols,vals)


    function eval_all(Z)
        eval_f(Z)
        eval_g(Z,g)
        eval_grad_f(Z,grad_f)
        eval_jac_g(Z,:Structure,rows,cols,vals)
        eval_jac_g(Z,:vals,rows,cols,vals)
    end
    eval_all(Z)

    # SNOPT method
    results = DircolResults(solver,method)
    results.Z .= Z
    update_derivatives!(solver,results,method)
    get_traj_points!(solver,results,method)
    get_traj_points_derivatives!(solver,results,method)
    update_jacobians!(solver,results,method)
    J_snopt = cost(solver,results)
    g_snopt = collocation_constraints(solver,results,method)
    results.X_
    grad_f_snopt = cost_gradient(solver,results,method)
    jac_g_snopt = constraint_jacobian(solver,results,method)

    function eval_all_snopt(Z)
        results.Z .= Z
        update_derivatives!(solver,results,method)
        get_traj_points!(solver,results,method)
        get_traj_points_derivatives!(solver,results,method)
        update_jacobians!(solver,results,method)
        J_snopt = cost(solver,results)
        g_snopt = collocation_constraints(solver,results,method)
        grad_f_snopt = cost_gradient(solver,results,method)
        jac_g_snopt = constraint_jacobian(solver,results,method)
        return nothing
    end
    eval_all_snopt(Z)

    # @time eval_all(Z)
    # @time eval_all_snopt(Z)

    @test J_snopt == J
    @test g_snopt == g
    @test grad_f_snopt == grad_f

    dropzeros!(jac_g_snopt)
    dropzeros!(jac_g)
    @test nonzeros(jac_g_snopt) == nonzeros(jac_g)
end
test_ipopt_funcs(:midpoint)
test_ipopt_funcs(:trapezoid)
test_ipopt_funcs(:hermite_simpson)
test_ipopt_funcs(:hermite_simpson_separated)


# Try IPOPT
method = :hermite_simpson_separated
u_bnd = 10
dt = 0.01
solver = Solver(model,ConstrainedObjective(obj,u_max=u_bnd,u_min=-u_bnd),dt=dt,integration=:rk3_foh)
N, = get_N(solver.N,method)
U0 = zeros(m,N)
X0 = line_trajectory(solver,method)
Z = packZ(X0,U0)
@time sol, stats, prob = solve_dircol(solver,X0,U0,method=method,nlp=:ipopt)
@test norm(sol.X[:,end]-obj.xf) < 1e-6
