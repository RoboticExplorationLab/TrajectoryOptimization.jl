using TrajectoryOptimization
using Plots
using Base.Test

dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
opts.c1 = 1e-4
opts.c2 = 2.0
opts.mu_al_update = 100.0
opts.infeasible_regularization = 1.0
opts.eps_constraint = 1e-3
opts.eps = 1e-5
opts.iterations_outerloop = 25
opts.iterations = 100

obj_uncon = TrajectoryOptimization.Dynamics.pendulum![2]
obj_uncon.R[:] = [1e-2]
solver_foh = Solver(Dynamics.pendulum![1], obj_uncon, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(Dynamics.pendulum![1], obj_uncon, integration=:rk3, dt=dt, opts=opts)

U = rand(solver_foh.model.m, solver_foh.N)

sol_zoh = TrajectoryOptimization.solve(solver_zoh,U)
sol_foh = TrajectoryOptimization.solve(solver_foh,U)

### test final state of foh solve
@test norm(solver_foh.obj.xf - sol_foh.X[:,end]) < 1e-3
sol_foh.X[:,end] - solver_foh.obj.xf
plot(sol_foh.X')
plot(sol_foh.U')
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

### test that control constraints work with foh
u_min = -2.0
u_max = 2.0
obj_uncon.R[:] = [5e-1]
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max) # constrained objective

solver_foh_con = Solver(Dynamics.pendulum![1], obj_con, integration=:rk3_foh, dt=dt, opts=opts)
sol_foh_con = TrajectoryOptimization.solve(solver_foh_con,U)
plot(sol_foh_con.X')
plot(sol_foh_con.U')
@test norm(sol_foh_con.X[:,end] - solver_foh_con.obj.xf) < 1e-3
###

### test that state and control constraints work with foh
u_min = -3
u_max = 3
x_min = [-10; -10]
x_max = [10; 10]
obj_con2 = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(Dynamics.pendulum![1], obj_con2, integration=:rk3_foh, dt=dt, opts=opts)
sol_foh_con2 = TrajectoryOptimization.solve(solver_foh_con2,U)
plot(sol_foh_con2.X')
plot(sol_foh_con2.U')
sol_foh_con2.X[:,end]
@test norm(sol_foh_con2.X[:,end] - solver_foh_con2.obj.xf) < 1e-3
###

# ### test infeasible start with foh
# ## Set up pendulum for infeasible tests
# n = 2 # number of pendulum states
# m = 1 # number of pendulum controls
# model! = Model(Dynamics.pendulum_dynamics!,n,m) # inplace dynamics model
#
# opts = SolverOptions()
# opts.square_root = false
# opts.verbose = true
# opts.cache=true
# opts.c1=1e-4
# opts.c2=2.0
# opts.mu_al_update = 100.0
# opts.infeasible_regularization = 1.0
# opts.eps_constraint = 1e-3
# opts.eps = 1e-5
# opts.iterations_outerloop = 250
# opts.iterations = 1000
#
# # Constraints
# u_min = -2
# u_max = 2
# x_min = [-10;-10]
# x_max = [10; 10]
# obj_uncon = Dynamics.pendulum[2]
# obj_uncon.R[:] = [1e-2]
# obj = ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)
#
# solver = Solver(model!,obj,integration=:rk3_foh,dt=0.1,opts=opts)
#
# # test linear interpolation for state trajectory
# X_interp = line_trajectory(solver.obj.x0,solver.obj.xf,solver.N)
# U = ones(solver.model.m,solver.N)
#
# results = solve(solver,X_interp,U)
#
# x = 2
# plot(results.X',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
# plot(results.U',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")
# println(results.X[:,end])
# # trajectory_animation(results,filename="infeasible_start_state.gif",fps=5)
# # trajectory_animation(results,traj="control",filename="infeasible_start_control.gif",fps=5)
# idx = find(x->x==2,results.iter_type)
# plot(results.result[end].X')
#
# plot(results.result[idx[1]-1].U',color="green")
# plot!(results.result[idx[1]].U',color="red")
#
# # confirm that control output from infeasible start is a good warm start for constrained solve
# @test norm(results.result[idx[1]-1].U[1,:]-results.result[idx[1]].U[1,:]) < 1e-3
# tmp = ConstrainedResults(solver.model.n,solver.model.m,size(results.result[1].C,1),solver.N)
# tmp.U[:,:] = results.result[idx[1]-1].U[1,:]
# tmp2 = ConstrainedResults(solver.model.n,solver.model.m,size(results.result[1].C,1),solver.N)
# tmp2.U[:,:] = results.result[end].U
#
# rollout!(tmp,solver)
# plot(tmp.X')
#
# rollout!(tmp2,solver)
# plot!(tmp2.X')
#
# # confirm that state trajectory from infeasible start is similar to the unconstrained solve
# @test norm(tmp.X' - tmp2.X') < 5.0
