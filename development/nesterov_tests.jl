using LinearAlgebra, Plots

#########################
##  Double Integrator  ##
#########################

model,obj = Dynamics.double_integrator

solver = Solver(model,obj,N=51)
n,m,N = get_sizes(solver)
U0 = ones(m,N-1)

solver.opts.verbose = true
res,stats = solve(solver,U0)
plot(to_array(res.U)')

opts = SolverOptions()
tol = 1e-8
opts.cost_tolerance = tol
opts.constraint_tolerance = tol
opts.cost_tolerance_intermediate = tol

# Constrained
obj_con = ConstrainedObjective(obj,u_min=-0.5,u_max=0.5)
solver = Solver(model,obj_con,N=51, opts=opts)
solver.opts.verbose = true
solver.opts.outer_loop_update_type = :default
solver.opts.penalty_update_frequency = 1
solver.opts.use_nesterov = true
res,stats = solve(solver,U0)
stats["iterations"]

solver = Solver(model,obj_con,N=51, opts=opts)
solver.state.penalty_only
solver.opts.verbose = true
solver.opts.outer_loop_update_type = :feedback
solver.opts.penalty_update_frequency = 1
solver.opts.constraint_decrease_ratio = 0.25
solver.opts.constraint_tolerance_coarse = 1e-6
solver.opts.use_nesterov
res_m,stats_m = solve(solver,U0)
solver.state.penalty_only
stats_m["iterations"]


stats_m["cost"][end] - stats["cost"][end]


#########################
##     Pendulum        ##
#########################

model,obj = Dynamics.pendulum
costfun = copy(obj.cost)
costfun.Qf .*= 0
xf = [pi; 0]
obj_con = ConstrainedObjective(costfun,obj.tf,obj.x0,xf)
obj_con2 = Dynamics.pendulum_constrained[2]

opts = SolverOptions()
tol = 1e-8
opts.cost_tolerance = tol
opts.constraint_tolerance = tol
opts.cost_tolerance_intermediate = 1e-4
opts.constraint_tolerance_coarse = 1e-2
opts.iterations_outerloop = 200

solver = Solver(model,obj_con,N=101,opts=opts)
solver.opts.verbose = true
n,m,N = get_sizes(solver)
U0 = ones(m,N-1)

solver.opts.outer_loop_update_type = :default
solver.opts.penalty_update_frequency = 1
solver.opts.use_nesterov = true
res,stats = solve(solver,U0)
stats["iterations"]
# no nesterov: 187
#    nesterov: 167

solver = Solver(model,obj_con,N=101,opts=opts)
TrajectoryOptimization.get_num_terminal_constraints(solver)
solver.opts.outer_loop_update_type = :feedback
solver.opts.penalty_update_frequency = 1
solver.opts.verbose = false
solver.opts.constraint_decrease_ratio = 0.25
solver.opts.use_nesterov = false
res_m,stats_m = solve(solver,U0)
stats_m["iterations"]
res_m.X[N]
# no nesterov: 176
#    nesterov: 176

stats_m["cost"][end] - stats["cost"][end]

plot(to_array(res.X)')
plot!(to_array(res_m.X)',linewidth=2)

########################
##     Dubins Car     ##
########################

model,obj = Dynamics.dubinscar
n,m = model.n, model.m
dt = 0.1

opts = SolverOptions()
opts.cost_tolerance = 1e-8
opts.constraint_tolerance = 1e-8
opts.cost_tolerance_intermediate = 1e-3

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]
obj_con = ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

solver = Solver(model,obj_con,N=51,opts=opts)
n,m,N = get_sizes(solver)
U0 = ones(m,N-1)

solver.opts.outer_loop_update_type = :default
solver.opts.verbose = true
solver.opts.use_nesterov = false
res,stats = solve(solver,U0)
stats["iterations"]

solver = Solver(model,obj_con,N=51,opts=opts)
solver.state.penalty_only
solver.opts.outer_loop_update_type = :feedback
solver.opts.constraint_decrease_ratio = 0.9
solver.opts.verbose = true
solver.opts.constraint_tolerance_coarse = 1e-4
solver.opts.use_nesterov = true
res_a,stats_a = solve(solver,U0)
stats_a["iterations"]

(stats["cost"][end] - stats_a["cost"][end])

plot(to_array(res.X)')
plot!(to_array(res_a.X)',linewidth=2)

########################
## Obstacle Avoidance ##
########################
using TrajectoryOptimization: circle_constraint, plot_obstacles, plot_trajectory!


x0 = [2.5;2.5;0.]
xf = [7.5;2.5;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

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

# plt = plot(title="Escape",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
# plot_obstacles(circles,:red)
# display(plt)

function cI(c,x,u)
    for i = 1:n_circles
        c[i] = circle_constraint(x,circles[i][1],circles[i][2],circles[i][3])
    end
    c
end

obj_con_obstacles = TrajectoryOptimization.ConstrainedObjective(obj,cI=cI)

opts = SolverOptions()
opts.verbose = false
opts.cost_tolerance = 1e-6
opts.cost_tolerance_intermediate = 1e-5
opts.constraint_tolerance = 1e-6
opts.R_infeasible = 1.0

solver = Solver(model, obj_con_obstacles, dt=dt, opts=opts)
n,m,N = get_sizes(solver)

# -Initial state and control trajectories
X_guess = [2.5 2.5 0.;4. 5. .785;5. 6.25 0.;7.5 6.25 -.261;9 5. -1.57;7.5 2.5 0.]
X0 = TrajectoryOptimization.interp_rows(N,tf,Array(X_guess'))
# X0 = line_trajectory(solver_uncon)
U0 = rand(m,N)

solver = Solver(model, obj_con_obstacles, dt=dt, opts=opts)
solver.opts.use_nesterov = true
solver.opts.outer_loop_update_type = :default
res, stats = TrajectoryOptimization.solve(solver,X0,U0)
stats["iterations"]
# w/o nesterov: 113
# w/  nesterov:  87

solver = Solver(model, obj_con_obstacles, dt=dt, opts=opts)
solver.opts.outer_loop_update_type = :feedback
solver.opts.use_nesterov = false
solver.opts.constraint_tolerance_coarse = 1e-4
res_a, stats_a = solve(solver,X0,U0)
stats_a["iterations"]
# w/o nesterov: 128
# w/  nesterov: 107

# w/o nesterov: 146
# w/  nesterov:  82


stats["cost"][end] - stats_a["cost"][end]

# Escape
plt = plot(title="Escape",aspect_ratio=:equal)
plot_obstacles(circles)
plot_trajectory!(to_array(res.X),width=2,color=:purple,label="Default",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot_trajectory!(to_array(res_a.X),width=2,color=:black,linestyle=:dash,label="Accelerated",aspect_ratio=:equal,xlim=[-1,11],ylim=[-1,11])
plot!(X0[1,:],X0[2,:],label="Infeasible Initialization",width=1,color=:black,linestyle=:dash)
display(plt)
