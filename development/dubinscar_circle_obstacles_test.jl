
function plot_obstacles(circles)
    for  circle in circles
        x,y,r = circle
        plot_circle!((x,y),r,color=:red,label="")
    end
end

function plot_trajectory!(X;kwargs...)
    plot!(X[1,:],X[2,:];kwargs...)
end

function plot_solution(X;kwargs...)
    p = plot(aspect_ratio=:equal,xlim=[0,12],ylim=[0,12])
    plot_obstacles(circles)
    plot_trajectory!(X; kwargs...)
end

### Solver Options ###
dt = 0.01
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
# opts.c1 = 1e-4
# opts.c2 = 2.0
opts.cost_tolerance_intermediate = 1e-1
opts.constraint_tolerance = 1e-6
opts.cost_tolerance = 1e-6
opts.iterations_outerloop = 100
opts.iterations = 1000
# opts.iterations_linesearch = 50
opts.constraint_decrease_ratio = 0.1
opts.outer_loop_update_type = :default
######################

### Set up model, objective, and solver ###
# Model
model, obj_uncon = TrajectoryOptimization.Dynamics.dubinscar!
n,m = model.n,model.m

# Objective
# -Circle obstacles, state, and control constraints
obj_uncon_obs = copy(obj_uncon)
obj_uncon_obs.x0[:] = [0;0;0]
obj_uncon_obs.xf[:] = [12;12;pi/2]
obj_uncon_obs.tf = 2.5
obj_uncon_obs.Qf[:,:] = 100.0*Diagonal(I,n)
obj_uncon_obs.R[:,:] = (1e-2)*Diagonal(I,m)

u_min = [-100; -100]
u_max = [100; 100]
x_min = [-20; -20; -100]
x_max = [20; 20; 100]

n_circles = 15
c = zeros(n_circles)
cI_obstacles, circles = generate_random_circle_obstacle_field(n_circles)

p = plot(aspect_ratio=:equal,xlim=[0,10],ylim=[0,10])
plot_obstacles(circles)
display(p)

# Solve with iLQR
obj_con_obs = TrajectoryOptimization.ConstrainedObjective(obj_uncon_obs,cI=cI_obstacles)# u_min=u_min, u_max=u_max,x_min=x_min, x_max=x_max,cI=cI) # constrained objective
solver = Solver(model, obj_con_obs, integration=:rk3, dt=dt, opts=opts)
solver.opts.verbose = false
solver.opts.resolve_feasible = false
n,m,N = get_sizes(solver)
X0 = line_trajectory(solver)
U = ones(m,N)
sol, stats = TrajectoryOptimization.solve(solver,X0,U)

plot_solution(to_array(sol.X),width=2,color=:black,label="iLQR")

# Solve with DIRCOL
res_d, stat_d = solve_dircol(solver,X0,U)

plot_trajectory!(res_d.X,width=2,color=:blue,linestyle=:dash,label="DIRCOL")

solver_uncon = Solver(model,obj_uncon_obs,dt=dt)
solver_uncon.opts.verbose = true
res_d, stat_d = solve_dircol(solver_uncon,X0,U)
plot(res_d.X[1,:],res_d.X[2,:])

plot_trajectory!

### Results ###
if opts.verbose
    println("Final state (foh): $(sol_foh_con_obs.X[:,end])")
    println("Final state (zoh): $(sol_zoh_con_obs.X[:,end])")

    println("Final cost (foh): $(sol_foh_con_obs.cost[sol_foh_con_obs.termination_index])")
    println("Final cost (zoh): $(sol_zoh_con_obs.cost[sol_zoh_con_obs.termination_index])")

    println("Termination index\n foh: $(sol_foh_con_obs.termination_index)\n zoh: $(sol_foh_con_obs.termination_index)")

    # plot(sol_foh_con_obs.U')
    # plot!(sol_zoh_con_obs.U')
    #
    # ## Plot obstacle field and trajectory
    # plot((solver_zoh_con_obs.obj.x0[1],solver_zoh_con_obs.obj.x0[2]),marker=(:circle,"red"),label="x0",xlim=(-1.1,11.1),ylim=(-1.1,11.1))
    # plot!((solver_zoh_con_obs.obj.xf[1],solver_zoh_con_obs.obj.xf[2]),marker=(:circle,"green"),label="xf")
    #
    # theta = linspace(0,2*pi,100)
    # for k = 1:n_circles
    #     x_circle = circles[3][k]*cos.(theta)
    #     y_circle = circles[3][k]*sin.(theta)
    #     plot!(x_circle+circles[1][k],y_circle+circles[2][k],color="red",width=2,fill=(100),legend=:none)
    # end
    #
    # plot!(sol_foh_con_obs.X[1,:],sol_foh_con_obs.X[2,:])
    # plot!(sol_zoh_con_obs.X[1,:],sol_zoh_con_obs.X[2,:])
end
###############
