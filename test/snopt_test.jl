# Test different derivative options
method = :hermite_simpson_separated
function check_grad_options(method)
    solver = Solver(model,obj,dt=dt)
    sol,stats = solve_dircol(solver,X0,U0,method=method,nlp=:snopt,grads=:none)
    @test vecnorm(sol.X[:,end]-obj.xf) < 1e-5

    sol2,stats = solve_dircol(solver,X0,U0,method=method,nlp=:snopt,grads=:auto)
    @test vecnorm(sol2.X[:,end]-obj.xf) < 1e-5
    @test norm(sol.X-sol2.X) < 5e-3
end

check_grad_options(:hermite_simpson_separated)
check_grad_options(:hermite_simpson)
check_grad_options(:trapezoid)
check_grad_options(:midpoint)



# Mesh refinement # TODO: mesh for ipopt
mesh = [0.5,0.2]
t1 = @elapsed sol,stats = solve_dircol(solver,X0,U0,nlp=:snopt)
t2 = @elapsed sol2,stats = solve_dircol(solver,X0,U0,mesh,)
@test vecnorm(sol.X[:,end]-obj.xf) < 1e-5
@test vecnorm(sol.X-sol.X2) < 1e-2
@test stats["info"][end] == "Finished successfully: optimality conditions satisfied"
