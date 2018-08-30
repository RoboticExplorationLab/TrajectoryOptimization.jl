
# Simple Pendulum
model = TrajectoryOptimization.Dynamics.pendulum![1]
obj = TrajectoryOptimization.Dynamics.pendulum[2]
obj = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-2, u_max=2)
dt = 0.1

x_opt, u_opt, fopt = dircol(model, obj, dt, method=:trapezoid, grads=:none)
# plot(x_opt')
# plot(u_opt')

# Setup
N = size(x_opt,2)
weights = TrajectoryOptimization.get_weights(:hermite_simpson_separated,N)*dt
pack = (model.n,model.m,N)
X,U = x_opt,u_opt
Z = TrajectoryOptimization.packZ(X,U)
f_aug! = TrajectoryOptimization.f_augmented!(model.f, model.n, model.m)
zdot = zeros(model.n)
F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)
usrfun, eval_f, eval_c, eval_ceq = TrajectoryOptimization.gen_usrfun(model,obj,dt, pack, :hermite_simpson_separated, grads=:auto)

# Constraints
eval_ceq(Z)
jacob_g = TrajectoryOptimization.constraint_jacobian(X,U,dt,:hermite_simpson_separated,F)
jacob_g_auto = ForwardDiff.jacobian(eval_ceq,Z)
@test errors = jacob_g_auto ≈ jacob_g

# Objective
grad_f = TrajectoryOptimization.cost_gradient(X,U,weights,obj)
grad_f_auto = ForwardDiff.gradient(eval_f,Z)
errors = grad_f - grad_f_auto
errors[end-10:end]
@test grad_f ≈ grad_f_auto
