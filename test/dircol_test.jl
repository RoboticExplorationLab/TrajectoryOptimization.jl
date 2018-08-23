
model = TrajectoryOptimization.Dynamics.pendulum![1]
obj = TrajectoryOptimization.Dynamics.pendulum[2]
obj = TrajectoryOptimization.ConstrainedObjective(obj, u_min=-2, u_max=2)
dt = 0.1

x_opt, u_opt, fopt = dircol(model, obj, dt, method=:trapezoid, grads=:none)
plot(x_opt')
plot(u_opt')

# Setup
weights = get_weights(:hermite_simpson_separated,N)
N = size(x_opt,2)
pack = (model.n,model.m,N)
X,U = x_opt,u_opt
Z = packZ(X,U)
f_aug! = f_augmented!(model!.f, model.n, model.m)
zdot = zeros(model.n)
F(z) = ForwardDiff.jacobian(f_aug!,zdot,z)
usrfun, eval_f, eval_g = gen_usrfun(model,obj,dt, pack, :hermite_simpson_separated, grads=:auto)

# Constraints
jacob_g = constraint_jacobian(X,U,dt,:hermite_simpson_separated,F)
jacob_g_auto = ForwardDiff.jacobian(eval_g,Z)
errors = jacob_g_auto == jacob_g

# Objective
grad_f = cost_gradient(X,U,weights,obj)
grad_f_auto = ForwardDiff.gradient(eval_f,Z)
errors = grad_f - grad_f_auto
errors[end-10:end]
grad_f ≈ grad_f_auto


# Cart Pole
