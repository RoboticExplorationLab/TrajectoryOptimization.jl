using TrajectoryOptimization
using Base.Test

dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache = false
opts.c1 = 0.15#1e-4
opts.c2 = 5.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-4
opts.iterations_outerloop = 250
opts.iterations = 100

obj_uncon = TrajectoryOptimization.Dynamics.pendulum![2]
solver_foh = Solver(Dynamics.pendulum![1], obj_uncon,dt=dt, integration=:rk3_foh, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(Dynamics.pendulum![1], obj_uncon, dt=dt,integration=:rk3, opts=opts)

U = ones(solver_foh.model.m, solver_foh.N)

sol_zoh = TrajectoryOptimization.solve(solver_zoh,U)
sol_foh = TrajectoryOptimization.solve(solver_foh,U)

sol_foh.X[:,end]

### test final state of foh solve
@test norm(solver_foh.obj.xf - sol_foh.X[:,end]) < 1e-3
###

### test that foh augmented dynamics works
n = 3
m = 2
fc! = TrajectoryOptimization.Dynamics.dubinscar![1].f
fc_aug! = TrajectoryOptimization.f_augmented!(fc!,m,n)
fd! = TrajectoryOptimization.rk3_foh(fc!,dt)
fd_aug! = TrajectoryOptimization.f_augmented_foh!(fd!,n,m)

x = ones(n)
u1 = ones(m)
u2 = ones(m)

@test norm(fd!(zeros(n),x,u1,u2) - fd_aug!(zeros(n+m+m+1),[x;u1;u2;dt])[1:n,1]) < 1e-5
###

### test that continuous dynamics Jacobians match known analytical solutions
solver_test = TrajectoryOptimization.Solver(Dynamics.dubinscar![1], obj_uncon, dt=dt,integration=:rk3_foh, opts=opts)

x = [0.0; 0.0; pi/4.0]
u = [2.0; 2.0]

Ac_known = [0.0 0.0 -sin(x[3])*u[1]; 0.0 0.0 cos(x[3])*u[1]; 0.0 0.0 0.0]
Bc_known = [cos(x[3]) 0.0; sin(x[3]) 0.0; 0.0 1.0]

Ac, Bc = solver_test.Fc(x,u)

@test norm((Ac_known - Ac)[:]) < 1e-5
@test norm((Bc_known - Bc)[:]) < 1e-5
###
