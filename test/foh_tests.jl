# Set random seed
Random.seed!(7)

### Solver Options ###
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
######################

### Simple Pendulum ###
obj_uncon_p = TrajectoryOptimization.Dynamics.pendulum![2]
model_p = Dynamics.pendulum![1]

## Unconstrained pendulum (foh)
solver_foh = TrajectoryOptimization.Solver(model_p, obj_uncon_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(model_p, obj_uncon_p, integration=:rk3, dt=dt, opts=opts)

U = ones(solver_foh.model.m, solver_foh.N)

sol_zoh, = TrajectoryOptimization.solve(solver_zoh,U)
sol_foh, = TrajectoryOptimization.solve(solver_foh,U)

# -compare foh and zoh state and control trajectories
# if opts.verbose
#     plot(sol_foh.X')
#     plot!(sol_zoh.X')
#
#     plot(sol_foh.U')
#     plot!(sol_zoh.U')
# end

# Test final state from foh solve
@test norm(solver_foh.obj.xf - sol_foh.X[:,end]) < 1e-3
##

## Control constraints pendulum (foh)
u_min = -2.0
u_max = 2.0
obj_con_p = TrajectoryOptimization.ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max) # constrained objective

solver_foh_con = Solver(model_p, obj_con_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con = Solver(model_p, obj_con_p, integration=:rk3, dt=dt, opts=opts)

U = ones(solver_foh_con.model.m, solver_foh_con.N)

sol_foh_con, = TrajectoryOptimization.solve(solver_foh_con,U)
sol_zoh_con, = TrajectoryOptimization.solve(solver_zoh_con,U)

# if opts.verbose
#     plot(sol_foh_con.X')
#     plot!(sol_zoh_con.X')
#     plot(sol_foh_con.U')
#     plot!(sol_zoh_con.U')
# end

# Test final state from foh solve
@test norm(sol_foh_con.X[:,end] - solver_foh_con.obj.xf) < 1e-3
##

## State and control constraints pendulum (foh)
u_min = -20
u_max = 6
x_min = [-10; -2]
x_max = [10; 6]
obj_con2_p = TrajectoryOptimization.ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(model_p, obj_con2_p, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con2 = Solver(model_p, obj_con2_p, integration=:rk3, dt=dt, opts=opts)

sol_foh_con2, = TrajectoryOptimization.solve(solver_foh_con2,U)
sol_zoh_con2, = TrajectoryOptimization.solve(solver_zoh_con2,U)

# if opts.verbose
#     plot(sol_foh_con2.X')
#     plot!(sol_zoh_con2.X')
#
#     plot(sol_foh_con2.U')
#     plot!(sol_zoh_con2.U')
# end

# Test final state from foh solve
@test norm(sol_foh_con2.X[:,end] - solver_foh_con2.obj.xf) < 1e-3
###

## Unconstrained infeasible start pendulum (foh)
solver_uncon_inf = Solver(model_p,obj_uncon_p,integration=:rk3_foh,dt=dt,opts=opts)

# -initial state and control trajectories
X_interp = line_trajectory(solver_uncon_inf.obj.x0,solver_uncon_inf.obj.xf,solver_uncon_inf.N)
U = ones(solver_uncon_inf.model.m,solver_uncon_inf.N)

results_inf, = solve(solver_uncon_inf,X_interp,U)

# if opts.verbose
#     plot(results_inf.X',title="Pendulum (Infeasible start with unconstrained control and states (inplace dynamics))",ylabel="x(t)")
#     plot(results_inf.U',title="Pendulum (Infeasible start with unconstrained control and states (inplace dynamics))",ylabel="u(t)")
#     println("Final state: $(results_inf.X[:,end])")
#     println("Final cost: $(results_inf.cost[end])")
# end
idx = findall(x->x==2,results_inf.iter_type)

# Test that infeasible control output is good warm start for dynamically constrained solve
@test norm(results_inf.result[idx[1]-1].U[1,:]-results_inf.result[idx[1]+1].U[1,:]) < 100.0 # TODO fix

# Test final (dynamically constrained) state from foh solve
@test norm(results_inf.X[:,end] - solver_uncon_inf.obj.xf) < 1e-3

# if opts.verbose
#     plot(results_inf.result[idx[1]-1].U',color="green")
#     plot!(results_inf.result[idx[1]+1].U',color="blue")
#     plot!(results_inf.result[end].U',color="red")
#     results_inf.result[idx[1]].U
#
#     # Confirm visually that control output from infeasible start is a good warm start for constrained solve
#     tmp = ConstrainedResults(solver_uncon_inf.model.n,solver_uncon_inf.model.m,size(results_inf.result[1].C,1),solver_uncon_inf.N)
#     tmp.U[:,:] = results_inf.result[idx[1]-1].U[1,:]
#     tmp2 = ConstrainedResults(solver_uncon_inf.model.n,solver_uncon_inf.model.m,size(results_inf.result[1].C,1),solver_uncon_inf.N)
#     tmp2.U[:,:] = results_inf.result[end].U
#
#     rollout!(tmp,solver_uncon_inf)
#     plot(tmp.X')
#     tmp.X[:,end]
#
#     rollout!(tmp2,solver_uncon_inf)
#     plot!(tmp2.X')
# end
##

## Infeasible start with constraints pendulum (foh)
u_min = -10
u_max = 10
x_min = [-10;-10]
x_max = [10; 10]

obj_con_p = ConstrainedObjective(obj_uncon_p, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max)

solver_con2 = Solver(model_p,obj_con_p,integration=:rk3_foh,dt=dt,opts=opts)

# -Linear interpolation for state trajectory
X_interp = line_trajectory(solver_con2.obj.x0,solver_con2.obj.xf,solver_con2.N)
U = ones(solver_con2.model.m,solver_con2.N)

results_inf2, = solve(solver_con2,X_interp,U)

# Test final state from foh solve
@test norm(results_inf2.X[:,end] - solver_con2.obj.xf) < 1e-3
#
# if opts.verbose
#     plot(results_inf2.X',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="x(t)")
#     plot(results_inf2.U',title="Pendulum (Infeasible start with constrained control and states (inplace dynamics))",ylabel="u(t)")
#     println("Final state: $(results_inf2.X[:,end])")
#     println("Final cost: $(results_inf2.cost[end])")
#     # trajectory_animation(results,filename="infeasible_start_state.gif",fps=5)
#     # trajectory_animation(results,traj="control",filename="infeasible_start_control.gif",fps=5)
# end
idx = findall(x->x==2,results_inf2.iter_type)

# if opts.verbose
#     plot(results_inf2.result[idx[1]-1].U',color="green")
#     plot!(results_inf2.result[idx[1]+1].U',color="red")
# end

# Confirm that control output from infeasible start is a good warm start for constrained solve
@test norm(results_inf2.result[idx[1]-1].U[1,:]-results_inf2.result[idx[1]].U[1,:]) < 1e-3

tmp = ConstrainedResults(solver_con2.model.n,solver_con2.model.m,size(results_inf2.result[1].C,1),solver_con2.N)
tmp.U[:,:] = results_inf2.result[idx[1]-1].U[1,:]
tmp2 = ConstrainedResults(solver_con2.model.n,solver_con2.model.m,size(results_inf2.result[1].C,1),solver_con2.N)
tmp2.U[:,:] = results_inf2.result[end].U

rollout!(tmp,solver_con2)
rollout!(tmp2,solver_con2)

# if opts.verbose
#     plot(tmp.X')
#     plot!(tmp2.X')
# end

# Confirm that state trajectory from infeasible start is similar to the unconstrained solve
@test norm(tmp.X' - tmp2.X') < 15.0
###################

#----------------#

### Dubins Car ###
obj_uncon_dc = TrajectoryOptimization.Dynamics.dubinscar![2]
model_dc = Dynamics.dubinscar![1]

# Unconstrained Dubins car (foh)
solver_foh = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3_foh, opts=opts)
solver_zoh = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3, opts=opts)

U = rand(solver_foh.model.m, solver_foh.N)

sol_zoh, = TrajectoryOptimization.solve(solver_zoh,U)
sol_foh, = TrajectoryOptimization.solve(solver_foh,U)

# if opts.verbose
#     plot(sol_foh.X[1,:],sol_foh.X[2,:])
#     plot!(sol_zoh.X[1,:],sol_zoh.X[2,:])
#
#     plot(sol_foh.U')
#     plot!(sol_zoh.U')
#
#     println("Final state (foh): $(sol_foh.X[:,end])")
#     println("Final state (zoh): $(sol_zoh.X[:,end])")
#
#     println("Final state cost (foh): $(sol_foh.cost[sol_foh.termination_index])")
#     println("Final state cost (zoh): $(sol_zoh.cost[sol_zoh.termination_index])")
# end

# Test final state from foh solve
@test norm(sol_foh.X[:,end] - solver_foh.obj.xf) < 1e-3
##

## State and control constraints Dubins car (foh)
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

# if opts.verbose
#     plot(sol_foh_con2.X[1,:],sol_foh_con2.X[2,:])
#     plot!(sol_zoh_con2.X[1,:],sol_zoh_con2.X[2,:])
#
#     println("Final state (foh): $(sol_foh_con2.X[:,end])")
#     println("Final state (zoh): $(sol_zoh_con2.X[:,end])")
#
#     println("Final cost (foh): $(sol_foh_con2.cost[sol_foh_con2.termination_index])")
#     println("Final cost (zoh): $(sol_zoh_con2.cost[sol_zoh_con2.termination_index])")
#
#     plot(sol_foh_con2.U')
#     plot!(sol_zoh_con2.U')
# end

# Test final state from foh solve
@test norm(sol_foh_con2.X[:,end] - solver_foh_con2.obj.xf) < 1e-3
##

## Infeasible start with state and control constraints Dubins car (foh)
u_min = [-1; -1]
u_max = [100; 100]
x_min = [0; -100; -100]
x_max = [1.0; 100; 100]

obj_con2_dc = TrajectoryOptimization.ConstrainedObjective(obj_uncon_dc, u_min=u_min, u_max=u_max, x_min=x_min, x_max=x_max) # constrained objective

solver_foh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con2 = Solver(model_dc, obj_con2_dc, integration=:rk3, dt=dt, opts=opts)

# -initial control and state trajectoires
U = 5*rand(solver_foh_con2.model.m,solver_foh_con2.N)
X_interp = line_trajectory(solver_foh_con2)

sol_foh_con2, = TrajectoryOptimization.solve(solver_foh_con2,X_interp,U)
sol_zoh_con2, = TrajectoryOptimization.solve(solver_zoh_con2,X_interp,U)

# if opts.verbose
#     plot(sol_foh_con2.X[1,:],sol_foh_con2.X[2,:])
#     plot!(sol_zoh_con2.X[1,:],sol_zoh_con2.X[2,:])
#
#     println("Final state (foh): $(sol_foh_con2.X[:,end])")
#     println("Final state (zoh): $(sol_zoh_con2.X[:,end])")
#
#     println("Final cost (foh): $(sol_foh_con2.cost[sol_foh_con2.termination_index])")
#     println("Final cost (zoh): $(sol_zoh_con2.cost[sol_zoh_con2.termination_index])")
#
#     plot(sol_foh_con2.U')
#     plot!(sol_zoh_con2.U')
# end

@test norm(sol_foh_con2.X[:,end] - solver_foh_con2.obj.xf) < 1e-3
##
