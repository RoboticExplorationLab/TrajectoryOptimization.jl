using TrajectoryOptimization
using Plots

### Solver options ###
opts = SolverOptions()
opts.square_root = false
opts.verbose = true
opts.cache=true
# opts.c1=1e-4
opts.c2=2.0
# opts.mu_al_update = 10.0
opts.eps_constraint = 1e-3
opts.eps_intermediate = 1e-3
opts.eps = 1e-3
opts.outer_loop_update = :uniform
opts.τ = 0.1
# opts.iterations_outerloop = 250
# opts.iterations = 1000
######################

### Set up model, objective, solver ###
# Model
n = 13 # states (quadrotor w/ quaternions)
m = 4 # controls
model! = Model(Dynamics.quadrotor_dynamics!,n,m)
model_euler = Model(Dynamics.quadrotor_dynamics_euler!,12,m)


# Objective and constraints
Qf = 100.0*eye(n)
Q = (0.01)*eye(n)
R = (0.01)*eye(m)
tf = 5.0
dt = 0.05

Qf_euler = 100.0*eye(12)
Q_euler = (0.1)*eye(12)

# -initial state
x0 = zeros(n)
quat0 = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
x0[4:7] = quat0
x0

x0_euler = zeros(12)

# -final state
xf = zeros(n)
xf[1:3] = [20.0;20.0;0.0] # xyz position
quatf = eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
xf[4:7] = quatf
xf

xf_euler = zeros(12)
xf_euler[1:3] = [20.0;20.0;0.0]

# -control limits
u_min = -10.0
u_max = 10.0

# -obstacles
quad_radius = 3.0
sphere_radius = 1.0

# x0[1:3] = -1.*ones(3)
# xf[1:3] = 11.0*ones(3)
# n_spheres = 3
# spheres = ([5.0;7.0;3.0],[5.0;7.0;3.0],[5.0;7;3.0],[sphere_radius;sphere_radius;sphere_radius])
# function cI(x,u)
#     [sphere_constraint(x,spheres[1][1],spheres[2][1],spheres[3][1],spheres[4][1]+quad_radius);
#      sphere_constraint(x,spheres[1][2],spheres[2][2],spheres[3][2],spheres[4][2]+quad_radius);
#      sphere_constraint(x,spheres[1][3],spheres[2][3],spheres[3][3],spheres[4][3]+quad_radius)]
# end

n_spheres = 3
spheres = ([5.0;9.0;15.0;],[5.0;9.0;15.0],[0.0;0.0;0.0],[sphere_radius;sphere_radius;sphere_radius])
function cI(x,u)
    [sphere_constraint(x,spheres[1][1],spheres[2][1],spheres[3][1],spheres[4][1]+quad_radius);
     sphere_constraint(x,spheres[1][2],spheres[2][2],spheres[3][2],spheres[4][2]+quad_radius);
     sphere_constraint(x,spheres[1][3],spheres[2][3],spheres[3][3],spheres[4][3]+quad_radius);
     -x[3]]
end

# -constraint that quaternion should be unit
function cE(x,u)
    [x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2 - 1.0]
end

obj_uncon = UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_uncon_euler = UnconstrainedObjective(Q_euler, R, Qf_euler, tf, x0_euler, xf_euler)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, cI=cI, cE = cE)
obj_con_euler = TrajectoryOptimization.ConstrainedObjective(obj_uncon_euler, u_min=u_min, u_max=u_max,cI=cI)

# Solver
solver_uncon = Solver(model!,obj_uncon,integration=:rk3_foh,dt=dt,opts=opts)
solver = Solver(model!,obj_con,integration=:rk4,dt=dt,opts=opts)

solver_uncon_euler = Solver(model_euler,obj_uncon_euler,integration=:rk4,dt=dt,opts=opts)
solver_euler = Solver(model_euler,obj_con_euler,integration=:rk4,dt=dt,opts=opts)


# - Initial control and state trajectories
U = ones(solver.model.m, solver.N)
# X_interp = line_trajectory(solver)
# X_interp = line_trajectory_quadrotor_quaternion(solver.obj.x0,solver.obj.xf,solver.N)::Array{Float64,2}
##################

### Solve ###
results_uncon, stats_uncon = solve(solver_uncon,U)
# results,stats = solve(solver,U)

# plot(log.(results_uncon.cost[1:results_uncon.termination_index]))
