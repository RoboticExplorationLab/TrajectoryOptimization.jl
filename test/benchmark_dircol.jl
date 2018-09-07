using TrajectoryOptimization

model, obj0 = Dynamics.cartpole_analytical


obj = copy(obj0)
obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 50
x_bnd = [0.6,Inf,Inf,Inf]
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)
dt = 0.05

N,dt = TrajectoryOptimization.calc_N(obj.tf, dt)


println("Hermite Simpson")
solver = Solver(model,obj,dt=dt)
N = solver.N
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)
@time X,U = solve_dircol(solver,X0,U0,method=:hermite_simpson,grads=:quad)

solver = Solver(model,obj,dt=dt)
println("Trapezoid (no gradients)")
@time X,U = solve_dircol(solver,X0,U0,method=:trapezoid,grads=:none)
println("Trapezoid")
@time X,U = solve_dircol(solver,X0,U0,method=:trapezoid,grads=:quad)
println("HS Separated (no gradients)")
@time X,U = solve_dircol(solver,X0,U0,method=:hermite_simpson_separated,grads=:none)
println("HS Separated")
@time X,U = solve_dircol(solver,X0,U0,method=:hermite_simpson_separated,grads=:quad)
println("Hermite Simpson (no gradients)")
@time X,U = solve_dircol(solver,X0,U0,method=:hermite_simpson,grads=:none)
println("Hermite Simpson")
@time X,U = solve_dircol(solver,X0,U0,method=:hermite_simpson,grads=:quad)

# using Juno
# @profiler solve_dircol(solver,X0,U0,method=:hermite_simpson_separated,grads=:none)
