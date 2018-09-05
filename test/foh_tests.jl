using TrajectoryOptimization
using Plots
using Base.Test

srand(7)
#### Solver setup
dt = 0.1
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = false
opts.cache = true
# opts.c1 = 1e-4
# opts.c2 = 2.0
# opts.mu_al_update = 10.0
# opts.eps_constraint = 1e-3
# opts.eps = 1e-5
# opts.iterations_outerloop = 100
# opts.iterations = 1000
# opts.iterations_linesearch = 50
###

#### Systems
obj_uncon_dc = TrajectoryOptimization.Dynamics.dubinscar![2]
model_dc = Dynamics.dubinscar![1]

obj_uncon_p = TrajectoryOptimization.Dynamics.pendulum![2]
model_p = Dynamics.pendulum![1]
####

### Unconstrained dubins car foh
# obj_uncon_dc.R[:,:] = (1e-1)*eye(2)
solver_foh = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3_foh, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3, opts=opts)

U = rand(solver_foh.model.m, solver_foh.N)

println("zoh solve:")
sol_zoh, = TrajectoryOptimization.solve(solver_zoh,U)
println("foh solve:")
sol_foh, = TrajectoryOptimization.solve(solver_foh,U)

plot(sol_foh.cost[1:sol_foh.termination_index])
plot!(sol_zoh.cost[1:sol_zoh.termination_index])

# plot(log.(sol_foh.cost[1:sol_foh.termination_index]))
# plot!(log.(sol_zoh.cost[1:sol_zoh.termination_index]))

plot(sol_foh.X[1,:],sol_foh.X[2,:])
plot!(sol_zoh.X[1,:],sol_zoh.X[2,:])

plot(sol_foh.U')
plot!(sol_zoh.U')

println("Final state (foh): $(sol_foh.X[:,end])")
println("Final state (zoh): $(sol_zoh.X[:,end])")

println("Final state cost (foh): $(sol_foh.cost[sol_foh.termination_index])")
println("Final state cost (zoh): $(sol_zoh.cost[sol_zoh.termination_index])")

sol_foh.U[:,end-1]
@test norm(sol_foh.X[:,end] - solver_foh.obj.xf) < 1e-3
###

### Unconstrained pendulum foh
solver_foh = TrajectoryOptimization.Solver(model_p, obj_uncon_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(model_p, obj_uncon_p, integration=:rk3, dt=dt, opts=opts)

U = rand(solver_foh.model.m, solver_foh.N)

sol_zoh, = TrajectoryOptimization.solve(solver_zoh,U)
sol_foh, = TrajectoryOptimization.solve(solver_foh,U)

# plot(log.(sol_zoh.cost[1:sol_zoh.termination_index]))
# plot!(log.(sol_foh.cost[1:sol_foh.termination_index]))

plot(sol_foh.X')
plot!(sol_zoh.X')

plot(sol_foh.U')
plot!(sol_zoh.U')

@test norm(solver_foh.obj.xf - sol_foh.X[:,end]) < 1e-3 # test final state of foh solve
##

### Control constraints with foh (pendulum)
u_min = -2.0
u_max = 2.0
# obj_uncon_p.R[:] = [1e-1]
obj_con_p = TrajectoryOptimization.ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max) # constrained objective

solver_foh_con = Solver(model_p, obj_con_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con = Solver(model_p, obj_con_p, integration=:rk3, dt=dt, opts=opts)

U = rand(solver_foh_con.model.m, solver_foh_con.N)

sol_foh_con, = TrajectoryOptimization.solve(solver_foh_con,U)
sol_zoh_con, = TrajectoryOptimization.solve(solver_zoh_con,U)

plot(sol_foh_con.X')
plot!(sol_zoh_con.X')
plot(sol_foh_con.U')
plot!(sol_zoh_con.U')

@test norm(sol_foh_con.X[:,end] - solver_foh_con.obj.xf) < 1e-3
###

### State and control constraints with foh (pendulum)
u_min = -20
u_max = 6
x_min = [-10; -2]
x_max = [10; 6]
obj_con2_p = TrajectoryOptimization.ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(model_p, obj_con2_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con2 = Solver(model_p, obj_con2_p, integration=:rk3, dt=dt, opts=opts)

sol_foh_con2, = TrajectoryOptimization.solve(solver_foh_con2,U)
sol_zoh_con2, = TrajectoryOptimization.solve(solver_zoh_con2,U)

plot(sol_foh_con2.X')
plot!(sol_zoh_con2.X')

plot(sol_foh_con2.U')
plot!(sol_zoh_con2.U')

# plot((sol_zoh_con2.cost[1:sol_zoh_con2.termination_index]))
# plot!((sol_foh_con2.cost[1:sol_foh_con2.termination_index]))

sol_foh_con2.X[:,end]
@test norm(sol_foh_con2.X[:,end] - solver_foh_con2.obj.xf) < 1e-3
###

### State and control constraints with foh (dubins car)
u_min = [-1; -1]
u_max = [100; 100]
x_min = [0; -100; -100]
x_max = [1.0; 100; 100]

obj_con2_dc = TrajectoryOptimization.ConstrainedObjective(obj_uncon_dc, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3, dt=dt, opts=opts)

U = 5*rand(solver_foh_con2.model.m,solver_foh_con2.N)

sol_foh_con2, = TrajectoryOptimization.solve(solver_foh_con2,U)
sol_zoh_con2, = TrajectoryOptimization.solve(solver_zoh_con2,U)

plot(sol_foh_con2.X[1,:],sol_foh_con2.X[2,:])
plot!(sol_zoh_con2.X[1,:],sol_zoh_con2.X[2,:])

println("Final state (foh): $(sol_foh_con2.X[:,end])")
println("Final state (zoh): $(sol_zoh_con2.X[:,end])")

println("Final cost (foh): $(sol_foh_con2.cost[sol_foh_con2.termination_index])")
println("Final cost (zoh): $(sol_zoh_con2.cost[sol_zoh_con2.termination_index])")

plot(sol_foh_con2.U')
plot!(sol_zoh_con2.U')

# plot(log.(sol_zoh_con2.cost[1:sol_zoh_con2.termination_index]))
# plot!(log.(sol_foh_con2.cost[1:sol_foh_con2.termination_index]))

@test norm(sol_foh_con2.X[:,end] - solver_foh_con2.obj.xf) < 1e-3
# ###


### Unconstrained infeasible start foh (pendulum)
solver_uncon_inf = Solver(model_p,obj_uncon_p,integration=:rk3_foh,dt=dt,opts=opts)

X_interp = line_trajectory(solver_uncon_inf.obj.x0,solver_uncon_inf.obj.xf,solver_uncon_inf.N)
U = rand(solver_uncon_inf.model.m,solver_uncon_inf.N)

results_inf, = solve(solver_uncon_inf,X_interp,U)

plot(results_inf.X',title="Pendulum (Infeasible start with unconstrained control and states (inplace dynamics))",ylabel="x(t)")
plot(results_inf.U',title="Pendulum (Infeasible start with unconstrained control and states (inplace dynamics))",ylabel="u(t)")
println("Final state: $(results_inf.X[:,end])")
println("Final cost: $(results_inf.cost[end])")

idx = find(x->x==2,results_inf.iter_type)

# # test that infeasible control output is good warm start for dynamically constrained solve
@test norm(results_inf.result[idx[1]-1].U[1,:]-results_inf.result[idx[1]+1].U[1,:]) < 100.0 # TODO fix
@test norm(results_inf.X[:,end] - solver_uncon_inf.obj.xf) < 1e-3

plot(results_inf.result[idx[1]-1].U',color="green")
plot!(results_inf.result[idx[1]+1].U',color="blue")
plot!(results_inf.result[end].U',color="red")
results_inf.result[idx[1]].U

# confirm that control output from infeasible start is a good warm start for constrained solve
tmp = ConstrainedResults(solver_uncon_inf.model.n,solver_uncon_inf.model.m,size(results_inf.result[1].C,1),solver_uncon_inf.N)
tmp.U[:,:] = results_inf.result[idx[1]-1].U[1,:]
tmp2 = ConstrainedResults(solver_uncon_inf.model.n,solver_uncon_inf.model.m,size(results_inf.result[1].C,1),solver_uncon_inf.N)
tmp2.U[:,:] = results_inf.result[end].U

rollout!(tmp,solver_uncon_inf)
plot(tmp.X')
tmp.X[:,end]

rollout!(tmp2,solver_uncon_inf)
plot!(tmp2.X')
###

### Infeasible start with constraints foh (pendulum)
u_min = -10
u_max = 10
x_min = [-10;-10]
x_max = [10; 10]
# obj_con_p.R[:] = [1.0]
obj_con_p = ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

solver_con2 = Solver(model_p,obj_con_p,integration=:rk3_foh,dt=0.1,opts=opts)

# test linear interpolation for state trajectory
X_interp = line_trajectory(solver_con2.obj.x0,solver_con2.obj.xf,solver_con2.N)
U = rand(solver_con2.model.m,solver_con2.N)

results_inf2, = solve(solver_con2,X_interp,U)
@test norm(results_inf2.X[:,end] - solver_con2.obj.xf) < 1e-3

plot(results_inf2.X',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
plot(results_inf2.U',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")
println("Final state: $(results_inf2.X[:,end])")
println("Final cost: $(results_inf2.cost[end])")
# trajectory_animation(results,filename="infeasible_start_state.gif",fps=5)
# trajectory_animation(results,traj="control",filename="infeasible_start_control.gif",fps=5)
idx = find(x->x==2,results_inf2.iter_type)
plot(results_inf2.result[end].X')

plot(results_inf2.result[idx[1]-1].U',color="green")
plot!(results_inf2.result[idx[1]+1].U',color="red")

# confirm that control output from infeasible start is a good warm start for constrained solve
@test norm(results_inf2.result[idx[1]-1].U[1,:]-results_inf2.result[idx[1]].U[1,:]) < 1e-3
tmp = ConstrainedResults(solver_con2.model.n,solver_con2.model.m,size(results_inf2.result[1].C,1),solver_con2.N)
tmp.U[:,:] = results_inf2.result[idx[1]-1].U[1,:]
tmp2 = ConstrainedResults(solver_con2.model.n,solver_con2.model.m,size(results_inf2.result[1].C,1),solver_con2.N)
tmp2.U[:,:] = results_inf2.result[end].U

rollout!(tmp,solver_con2)
plot(tmp.X')

rollout!(tmp2,solver_con2)
plot!(tmp2.X')

# confirm that state trajectory from infeasible start is similar to the unconstrained solve
@test norm(tmp.X' - tmp2.X') < 10.0
###

### infeasible start with state and control constraints with foh (dubins car)
u_min = [-1; -1]
u_max = [100; 100]
x_min = [0; -100; -100]
x_max = [1.0; 100; 100]

opts.iterations = 300
opts.iterations_outerloop = 50

obj_con2_dc = TrajectoryOptimization.ConstrainedObjective(obj_uncon_dc, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3, dt=dt, opts=opts)

U = 5*rand(solver_foh_con2.model.m,solver_foh_con2.N)
X_interp = line_trajectory(solver_foh_con2)

sol_foh_con2, = TrajectoryOptimization.solve(solver_foh_con2,X_interp,U)
sol_zoh_con2, = TrajectoryOptimization.solve(solver_zoh_con2,X_interp,U)
plot(sol_foh_con2.X[1,:],sol_foh_con2.X[2,:])
plot!(sol_zoh_con2.X[1,:],sol_zoh_con2.X[2,:])

println("Final state (foh): $(sol_foh_con2.X[:,end])")
println("Final state (zoh): $(sol_zoh_con2.X[:,end])")

println("Final cost (foh): $(sol_foh_con2.cost[sol_foh_con2.termination_index])")
println("Final cost (zoh): $(sol_zoh_con2.cost[sol_zoh_con2.termination_index])")

plot(sol_foh_con2.U')
plot!(sol_zoh_con2.U')

@test norm(sol_foh_con2.X[:,end] - solver_foh_con2.obj.xf) < 1e-3
###

## MOVED TO jacobian_tests.jl
# ### foh augmented dynamics
# n_dc = 3
# m_dc = 2
# fc! = model_dc.f
# fc_aug! = TrajectoryOptimization.f_augmented!(fc!,m_dc,n_dc)
# fd! = TrajectoryOptimization.rk3_foh(fc!,dt)
# fd_aug! = TrajectoryOptimization.f_augmented_foh!(fd!,n_dc,m_dc)
#
# x = ones(n_dc)
# u1 = ones(m_dc)
# u2 = ones(m_dc)
#
# @test norm(fd!(zeros(n_dc),x,u1,u2) - fd_aug!(zeros(n_dc+m_dc+m_dc+1),[x;u1;u2;dt])[1:n_dc,1]) < 1e-5
# ###
#
# ### Continuous dynamics Jacobians match known analytical solutions
# solver_test = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3_foh, opts=opts)
#
# x = [0.0; 0.0; pi/4.0]
# u = [2.0; 2.0]
#
# Ac_known = [0.0 0.0 -sin(x[3])*u[1]; 0.0 0.0 cos(x[3])*u[1]; 0.0 0.0 0.0]
# Bc_known = [cos(x[3]) 0.0; sin(x[3]) 0.0; 0.0 1.0]
#
# Ac, Bc = solver_test.Fc(x,u)
#
# @test norm((Ac_known - Ac)[:]) < 1e-5
# @test norm((Bc_known - Bc)[:]) < 1e-5
# ###
