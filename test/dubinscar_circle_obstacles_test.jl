using TrajectoryOptimization
using Plots

dt = 0.01
opts = TrajectoryOptimization.SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache = true
# opts.c1 = 1e-4
# opts.c2 = 2.0
# opts.mu_al_update = 10.0
opts.eps_intermediate = 1e-2
opts.eps_constraint = 1e-5
opts.eps = 1e-5
opts.iterations_outerloop = 100
opts.iterations = 1000
# opts.iterations_linesearch = 50
opts.τ = 0.1
###

#### System
model, obj_uncon = TrajectoryOptimization.Dynamics.dubinscar!

### Circle obstacles, state, and control constraints for Dubins car
obj_uncon_obs = copy(obj_uncon)
obj_uncon_obs.x0[:] = [-1;-1;0]
obj_uncon_obs.xf[:] = [5;11;pi/2]
obj_uncon_obs.tf = 2.5
obj_uncon_obs.Qf[:,:] = 100.0*eye(3)

u_min = [-20; -100]
u_max = [10; 10]
x_min = [-20; -20; -100]
x_max = [20; 20; 100]

function circle_constraint(x,x0,y0,r)
	return -((x[1]-x0)^2 + (x[2]-y0)^2  - r^2)
end

# function generate_random_circle_obstacle_field(n_circles::Int64,x_rand::Float64=10.0,y_rand::Float64=10.0,r_rand::Float64=0.5)
#     x_origin = x_rand*rand(n_circles)
#     y_origin = y_rand*rand(n_circles)
#     r = r_rand*ones(n_circles)
#
#     function constraints(x,u)::Array
#         c = zeros(typeof(x[1]),n_circles)
#
#         for i = 1:n_circles
#             c[i] = circle_constraint(x,x_origin[i],y_origin[i],r[i])
#         end
#         c
#     end
#     constraints, (x_origin, y_origin, r)
# end

# n_circles = 5
# cI, circles = generate_random_circle_obstacle_field(n_circles,5.0)

n_circles = 3
circles = ([1.0;2.6;3.5],[1.25;5.0;7.5],[.25;.25;.25])
function cI(x,u)
    [circle_constraint(x,circles[1][1],circles[2][1],circles[3][1]);
     circle_constraint(x,circles[1][2],circles[2][2],circles[3][2]);
     circle_constraint(x,circles[1][3],circles[2][3],circles[3][3])]
end

obj_con_obs = TrajectoryOptimization.ConstrainedObjective(obj_uncon_obs, u_min=u_min, u_max=u_max,x_min=x_min, x_max=x_max,cI=cI) # constrained objective
solver_foh_con_obs = Solver(model, obj_con_obs, integration=:rk3_foh, dt=dt, opts=opts)
solver_zoh_con_obs = Solver(model, obj_con_obs, integration=:rk3, dt=dt, opts=opts)

U = 5*rand(solver_zoh_con_obs.model.m,solver_zoh_con_obs.N)

sol_foh_con_obs, = TrajectoryOptimization.solve(solver_foh_con_obs,U)
sol_zoh_con_obs, = TrajectoryOptimization.solve(solver_zoh_con_obs,U)

println("Final state (foh): $(sol_foh_con_obs.X[:,end])")
println("Final state (zoh): $(sol_zoh_con_obs.X[:,end])")

println("Final cost (foh): $(sol_foh_con_obs.cost[sol_foh_con_obs.termination_index])")
println("Final cost (zoh): $(sol_zoh_con_obs.cost[sol_zoh_con_obs.termination_index])")

println("Termination index\n foh: $(sol_foh_con_obs.termination_index)\n zoh: $(sol_foh_con_obs.termination_index)")

plot(sol_foh_con_obs.U')
plot!(sol_zoh_con_obs.U')

## Plot obstacle field and trajectory
plot((solver_zoh_con_obs.obj.x0[1],solver_zoh_con_obs.obj.x0[2]),marker=(:circle,"red"),label="x0",xlim=(-1.1,11.1),ylim=(-1.1,11.1))
plot!((solver_zoh_con_obs.obj.xf[1],solver_zoh_con_obs.obj.xf[2]),marker=(:circle,"green"),label="xf")

theta = linspace(0,2*pi,100)
for k = 1:n_circles
    x_circle = circles[3][k]*cos.(theta)
    y_circle = circles[3][k]*sin.(theta)
    plot!(x_circle+circles[1][k],y_circle+circles[2][k],color="red",width=2,fill=(100),legend=:none)
end

plot!(sol_foh_con_obs.X[1,:],sol_foh_con_obs.X[2,:])
plot!(sol_zoh_con_obs.X[1,:],sol_zoh_con_obs.X[2,:])
