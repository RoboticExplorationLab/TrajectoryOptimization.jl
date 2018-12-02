Random.seed!(7)

# Solver Options
dt = 0.01
integration = :rk4
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false

###################
## Parallel Park ##
###################

# Set up model, objective, and solver
model, = TrajectoryOptimization.Dynamics.dubinscar!
n, m = model.n,model.m

x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con_box = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

solver_uncon = Solver(model, obj, integration=integration, dt=dt, opts=opts)
solver_con_box = Solver(model, obj_con_box, integration=integration, dt=dt, opts=opts)

U0 = rand(solver_uncon.model.m,solver_uncon.N)

results_uncon, stats_uncon = TrajectoryOptimization.solve(solver_uncon,U0)
results_con_box, stats_con_box = TrajectoryOptimization.solve(solver_con_box,U0)

# plt = plot(title="Parallel Park")
# plot_trajectory!(to_array(results_uncon.X),width=2,color=:blue,label="",aspect_ratio=:equal,xlim=[-0.5,0.5],ylim=[-.25,1.25])
# display(plt)

# Parallel Park (boxed)
plt = plot(title="Parallel Park")#,aspect_ratio=:equal)
plot!(x_min[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(x_max[1]*ones(1000),collect(range(x_min[2],stop=x_max[2],length=1000)),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_min[2]*ones(1000),color=:red,width=2,label="")
plot!(collect(range(x_min[1],stop=x_max[1],length=1000)),x_max[2]*ones(1000),color=:red,width=2,label="")
plot_trajectory!(to_array(results_uncon.X),width=2,color=:blue,label="Unconstrained")
plot_trajectory!(to_array(results_con_box.X),width=2,color=:green,label="Constrained",legend=:bottomright)
display(plt)

########################
## Obstacle Avoidance ##
########################

x0 = [0.0;0.0;0.]
xf = [10.0;10.0;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

u_min = [-100.; -100.]
u_max = [100.; 100.]
x_min = [-Inf; -Inf; -Inf]
x_max = [Inf; Inf; Inf]

n_circles = 3
r = 1.0
circles = ((2.5,2.5,r),(5.,5.,r),(7.5,7.5,r))

function cI(c,x,u)
    for i = 1:n_circles
        c[i] = circle_constraint(x,circles[i][1],circles[i][2],circles[i][3])
    end
    c
end

obj_con_obstacles = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max,cI=cI)
obj_con_obstacles_control = TrajectoryOptimization.ConstrainedObjective(obj,u_min=u_min, u_max=u_max,x_min=x_min,x_max=x_max,cI=cI)

solver_uncon_obstacles = Solver(model, obj, integration=integration, dt=dt, opts=opts)
solver_con_obstacles = Solver(model, obj_con_obstacles, integration=integration, dt=dt, opts=opts)
solver_con_obstacles_control = Solver(model, obj_con_obstacles_control, integration=integration, dt=dt, opts=opts)

# -Initial state and control trajectories
U0 = rand(solver_uncon.model.m,solver_uncon.N)

results_uncon_obstacles, stats_uncon_obstacles = TrajectoryOptimization.solve(solver_uncon_obstacles,U0)
results_con_obstacles, stats_con_obstacles = TrajectoryOptimization.solve(solver_con_obstacles,U0)

# Obstacle Avoidance
plt = plot(title="Obstacle Avoidance")
plot_obstacles(circles)
plot_trajectory!(to_array(results_uncon_obstacles.X),width=2,color=:blue,label="Unconstrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(results_con_obstacles.X),width=2,color=:green,label="Constrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11],legend=:bottomright)
display(plt)

############
## Escape ##
############

x0 = [2.5;2.5;0.]
xf = [7.5;2.5;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)

r = 0.5
s1 = 30; s2 = 50; s3 = 15
circles = []
for i in range(0,stop=5,length=s1)
    push!(circles,(0.,i,r))
end
for i in range(0,stop=5,length=s1)
    push!(circles,(5.,i,r))
end
for i in range(0,stop=5,length=s1)
    push!(circles,(10.,i,r))
end
for i in range(0,stop=10,length=s2)
    push!(circles,(i,0.,r))
end
for i in range(0,stop=3,length=s3)
    push!(circles,(i,5.,r))
end
for i in range(5,stop=8,length=s3)
    push!(circles,(i,5.,r))
end

n_circles = 3*s1 + s2 + 2*s3

plt = plot(title="Escape",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_obstacles(circles,:red)
display(plt)

function cI(c,x,u)
    for i = 1:n_circles
        c[i] = circle_constraint(x,circles[i][1],circles[i][2],circles[i][3])
    end
    c
end

obj_con_obstacles = TrajectoryOptimization.ConstrainedObjective(obj,cI=cI)

solver_uncon_obstacles = Solver(model, obj, integration=integration, dt=dt, opts=opts)
solver_con_obstacles = Solver(model, obj_con_obstacles, integration=integration, dt=dt, opts=opts)
solver_con_obstacles.opts.R_infeasible = 1.0
# -Initial state and control trajectories
X_guess = [2.5 2.5 0.;3.75 5. .785;5. 6.25 0.;7.5 6.25 -.261;8.75 5. -1.57;7.5 2.5 0.]
X0 = interp_rows(solver_uncon.N,tf,Array(X_guess'))
# X0 = line_trajectory(solver_uncon)
U0 = rand(solver_uncon.model.m,solver_uncon.N)

results_uncon_obstacles, stats_uncon_obstacles = TrajectoryOptimization.solve(solver_uncon_obstacles,U0)
results_con_obstacles, stats_con_obstacles = TrajectoryOptimization.solve(solver_con_obstacles,U0)
results_con_obstacles_inf, stats_con_obstacles_inf = TrajectoryOptimization.solve(solver_con_obstacles,X0,U0)

# Escape
plt = plot(title="Escape",aspect_ratio=:equal)
plot_obstacles(circles)
plot_trajectory!(to_array(results_uncon_obstacles.X),width=2,color=:blue,label="Unconstrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(results_con_obstacles.X),width=2,color=:green,label="Constrained",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(results_con_obstacles_inf.X),width=2,color=:purple,label="Infeasible",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot!(X0[1,:],X0[2,:],label="Infeasible Initialization",width=1,color=:purple,linestyle=:dash)
display(plt)
