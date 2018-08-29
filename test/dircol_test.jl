
model = TrajectoryOptimization.Dynamics.pendulum![1]
obj = TrajectoryOptimization.Dynamics.pendulum[2]
obj = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-2, u_max=2)
dt = 0.1

x_opt, u_opt, fopt = dircol(model, obj, dt, method=:trapezoid, grads=:none)
plot(x_opt')
plot(u_opt')

# Setup
N = size(x_opt,2)
weights = get_weights(:hermite_simpson_separated,N)*dt
pack = (model.n,model.m,N)
X,U = x_opt,u_opt
Z = packZ(X,U)
f_aug! = f_augmented!(model.f, model.n, model.m)
zdot = zeros(model.n)
F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)
usrfun, eval_f, eval_g = gen_usrfun(model,obj,dt, pack, :hermite_simpson_separated, grads=:auto)

# Constraints
eval_g(Z)
jacob_g = constraint_jacobian(X,U,dt,:hermite_simpson_separated,F)
jacob_g_auto = ForwardDiff.jacobian(eval_g,Z)
errors = jacob_g_auto == jacob_g

# Objective
grad_f = cost_gradient(X,U,weights,obj)
grad_f_auto = ForwardDiff.gradient(eval_f,Z)
errors = grad_f - grad_f_auto
errors[end-10:end]
grad_f ≈ grad_f_auto


# Cart Pole (iLQR)
model, obj = Dynamics.cartpole!
obj.x0 = [0,0,0,0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 50
obj_con = ConstrainedObjective(obj,u_min = -u_bnd, u_max = u_bnd)
solver = Solver(model,obj_con,dt=0.1)
U = ones(1,solver.N)*10
X = line_trajectory(obj.x0, obj.xf, solver.N)
solver.opts.verbose = true
results = solve(solver,X,U)
plot(results.X[1:2,:]')
plot(results.U')
U_opt = copy(results.U)

TrajectoryOptimization.set_debug_level(:info)
x_opt,u_opt, f_opt = dircol(model, obj_con, 0.1, method=:hermite_simpson_separated, grads=:none)
plot(x_opt[1:2,:]')
plot(u_opt')


N = size(x_opt,2)
Z0 = get_initial_state(obj,N)
weights = get_weights(:hermite_simpson_separated,N)*dt
pack = (model.n,model.m,N)
X,U = x_opt,u_opt
Z = packZ(X,U)
f_aug! = f_augmented!(model.f, model.n, model.m)
zdot = zeros(model.n)
F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)
usrfun, eval_f, eval_c, eval_ceq = gen_usrfun(model,obj_con,dt, pack, :hermite_simpson_separated, grads=:auto)

eval_ceq(Z)
