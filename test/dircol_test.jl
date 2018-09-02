#***************#
#   CART POLE   #
#***************#
model, obj0 = Dynamics.cartpole_analytical

obj = copy(obj0)
obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 50
x_bnd = [0.6,Inf,Inf,Inf]
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd)
dt = 0.1

N,dt = TrajectoryOptimization.calc_N(obj.tf, dt)
U0 = ones(1,N)*1
X0 = line_trajectory(obj.x0, obj.xf, N)

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
solver = Solver(model,obj,dt=dt)
solver.opts.verbose = true
X,U = solve_dircol(solver,X0,U0,method=method,grads=:quad)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test size(X,2) == 2N-1

X2,U2 = solve_dircol(solver,X0,U0,method=method,grads=:auto)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test X ≈ X2

X3,U3 = solve_dircol(solver,X0,U0,method=method,grads=:none)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test vecnorm(X-X3) < 1e-3

method = :trapezoid
solver = Solver(model,obj,dt=dt)
X,U = solve_dircol(solver,X0,U0,method=method,grads=:quad)
@test vecnorm(X[:,end]-obj.xf) < 1e-5

X2,U2 = solve_dircol(solver,X0,U0,method=method,grads=:auto)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test X ≈ X2

X3,U3 = solve_dircol(solver,X0,U0,method=method,grads=:none)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test vecnorm(X-X3) < 1e-3


# Mesh refinement
mesh = [0.5,0.2]
t1 = @elapsed X,U = solve_dircol(solver,X0,U0)
t2 = @elapsed X2,U2 = solve_dircol(solver,X0,U0,mesh)
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test vecnorm(X2-X) < 1e-2
@test t2 < t1


# No initial guess
mesh = [0.5]
t1 = @elapsed X,U = solve_dircol(solver)
t2 = @elapsed X2,U2 = solve_dircol(solver,mesh)
@test vecnorm(X[:,end]-obj.xf) < 1e-5
@test vecnorm(X2[:,end]-obj.xf) < 1e-5
@test vecnorm(X2-X) < 1e-2
@test t2/2 < t1
