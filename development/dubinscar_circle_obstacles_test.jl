
function plot_obstacles(circles)
    for  circle in circles
        x,y,r = circle
        plot_circle!((x,y),r,color=:red,label="")
    end
end

function plot_trajectory!(X;kwargs...)
    plot!(X[1,:],X[2,:];kwargs...)
end

### Solver Options ###
dt = 0.01
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
# opts.c1 = 1e-4
# opts.c2 = 2.0
opts.cost_intermediate_tolerance = 1e-1
opts.constraint_tolerance = 1e-6
opts.cost_tolerance = 1e-6
opts.iterations_outerloop = 100
opts.iterations = 1000
# opts.iterations_linesearch = 50
opts.τ = 0.1
opts.outer_loop_update = :default
######################

### Set up model, objective, and solver ###
# Model
model, obj_uncon = TrajectoryOptimization.Dynamics.dubinscar!
n,m = model.n,model.m

# Objective
# -Circle obstacles, state, and control constraints
obj_uncon_obs = copy(obj_uncon)
obj_uncon_obs.x0[:] = [0;0;0]
obj_uncon_obs.xf[:] = [11;11;pi/2]
obj_uncon_obs.tf = 2.5
obj_uncon_obs.Qf[:,:] = 100.0*Diagonal(I,n)
obj_uncon_obs.R[:,:] = (1e-2)*Diagonal(I,m)

u_min = [-100; -100]
u_max = [100; 100]
x_min = [-20; -20; -100]
x_max = [20; 20; 100]

# n_circles = 3
# circles = ([1.0;2.6;3.5],[1.25;5.0;7.5],[.25;.25;.25])
# function cI(x,u)
#     [circle_constraint(x,circles[1][1],circles[2][1],circles[3][1]);
#      circle_constraint(x,circles[1][2],circles[2][2],circles[3][2]);
#      circle_constraint(x,circles[1][3],circles[2][3],circles[3][3])]
# end

n_circles = 15
c = zeros(n_circles)
cI, circles = generate_random_circle_obstacle_field(n_circles)

p = plot(aspect_ratio=:equal,xlim=[0,10],ylim=[0,10])
plot_obstacles(circles)
display(p)

obj_con_obs = TrajectoryOptimization.ConstrainedObjective(obj_uncon_obs,cI=cI)# u_min=u_min, u_max=u_max,x_min=x_min, x_max=x_max,cI=cI) # constrained objective
solver = Solver(model, obj_con_obs, integration=:rk3, dt=dt, opts=opts)

# -Initial state and control trajectories
X0 = line_trajectory(solver_zoh_con_obs)
U = ones(solver_zoh_con_obs.model.m,solver_zoh_con_obs.N)
# U = 5*rand(solver_zoh_con_obs.model.m,solver_zoh_con_obs.N)
#############################################

### Solve ###
sol, stats = TrajectoryOptimization.solve(solver,X_interp,U)
#############

p = plot(aspect_ratio=:equal,xlim=[0,10],ylim=[0,10])
plot_obstacles(circles)
plot_trajectory!(to_array(sol.X),width=2,color=:black,label="")


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
