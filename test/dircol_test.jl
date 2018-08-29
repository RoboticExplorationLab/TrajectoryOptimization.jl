
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
@test errors = jacob_g_auto == jacob_g

# Objective
grad_f = cost_gradient(X,U,weights,obj)
grad_f_auto = ForwardDiff.gradient(eval_f,Z)
errors = grad_f - grad_f_auto
errors[end-10:end]
@test grad_f ≈ grad_f_auto



# Cart Pole
state = Dynamics.cartpole_mechanism
model, obj = Dynamics.cartpole_analytical

obj.x0 = [0;0;0;0.]
obj.xf = [0.5;pi;0;0]
obj.tf = 2.
u_bnd = 50
x_bnd = [0.6,Inf,Inf,Inf]
dt = 0.1
obj_con = ConstrainedObjective(obj,u_min=-u_bnd, u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd)

# iLQR
solver = Solver(model,obj_con,dt=0.1)
U = ones(1,solver.N)*10
X = line_trajectory(obj.x0, obj.xf, solver.N)
solver.opts.verbose = true
@time results = solve(solver,X,U)
plot(results.X[1:2,:]')
plot(results.U')
U_opt = copy(results.U)


x_opt,u_opt, f_opt = dircol(model, obj_con, 0.1, method=:trapezoid, grads=:quad)
plot(x_opt[1:2,:]')
plot(u_opt')

# Compare
time_dircol = linspace(0,obj_con.tf,size(x_opt,2))
time_ilqr = linspace(0,obj_con.tf,size(results.X,2))
state_labels = ["position (m)" "angle (rad)"]
colors = [:blue :red]
plot(time_dircol,x_opt[1:2,:]',width=2, label=state_labels.*" (DIRCOL)",color=colors)
plot!(time_ilqr,results.X[1:2,:]',width=2,label=state_labels.*" (iLQR)",color=colors, style=:dash,
    xlabel="time (s)", ylabel="States",legend=:topleft)

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
